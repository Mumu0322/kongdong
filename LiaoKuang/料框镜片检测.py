#!/usr/bin/env python3
"""
lens_detection_standalone.py - 独立的镜片检测模块
不依赖ROS2，直接使用TensorRT进行推理
输入: BGR图像
输出: 所有检测到的镜片中心坐标列表 [(x1,y1), (x2,y2), ...]
"""

import argparse
import time
import struct
from pathlib import Path
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class LensDetector:
    """镜片检测器 - 纯视觉检测，返回镜片中心坐标"""

    def __init__(self, model_path, conf_threshold=0.15):
        """
        初始化检测器

        Args:
            model_path: TensorRT engine文件路径
            conf_threshold: 置信度阈值
        """
        self.conf_threshold = conf_threshold
        self._load_engine(model_path)

    def _load_engine(self, engine_path):
        """加载TensorRT引擎"""
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        # 读取并跳过ultralytics JSON头部
        with open(engine_path, 'rb') as f:
            data = f.read()
        json_len = struct.unpack('<I', data[:4])[0]
        trt_data = data[4 + json_len:]

        self.engine = runtime.deserialize_cuda_engine(trt_data)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # 分配输入输出缓冲区
        self.bindings = {}
        self.input_name = None
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self._trt_to_np_dtype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)

            host_buf = cuda.pagelocked_empty(shape, dtype)
            device_buf = cuda.mem_alloc(host_buf.nbytes)

            self.bindings[name] = {'host': host_buf, 'device': device_buf, 'shape': shape}

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        # 获取输入尺寸
        self.input_h = self.bindings[self.input_name]['shape'][2]
        self.input_w = self.bindings[self.input_name]['shape'][3]

    def _trt_to_np_dtype(self, trt_dtype):
        mapping = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT8: np.int8,
            trt.DataType.INT32: np.int32,
        }
        return mapping.get(trt_dtype, np.float32)

    def _preprocess(self, img_bgr):
        """预处理: BGR -> RGB, letterbox缩放, 归一化"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_w, self.input_h),
                                 interpolation=cv2.INTER_LINEAR)
        img_float = img_resized.astype(np.float32) / 255.0
        # HWC -> NCHW
        inp = np.ascontiguousarray(
            img_float.transpose(2, 0, 1)[np.newaxis, :]
        )
        return inp

    def _postprocess(self, outputs, orig_h, orig_w):
        """
        后处理: 解析检测结果，提取镜片中心坐标

        Args:
            outputs: 模型输出 {'output0': np, 'output1': np}
            orig_h: 原始图像高度
            orig_w: 原始图像宽度

        Returns:
            centers: 镜片中心坐标列表 [(x1,y1), (x2,y2), ...]
        """
        dets = outputs['output0'][0]   # (300, 38)
        protos = outputs['output1'][0]  # (32, 320, 320)

        # engine输入尺寸（从output1反推）
        infer_h = protos.shape[1] * 4   # 1280
        infer_w = protos.shape[2] * 4   # 1280

        scale_x = orig_w / infer_w
        scale_y = orig_h / infer_h

        centers = []

        for det in dets:
            conf = float(det[4])
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            mask_coef = det[6:38]   # 32个mask系数

            # 解码mask
            proto_flat = protos.reshape(32, -1)  # (32, 102400)
            mask_flat = mask_coef @ proto_flat   # (102400,)
            mask = mask_flat.reshape(protos.shape[1], protos.shape[2])
            mask = 1.0 / (1.0 + np.exp(-mask))   # sigmoid
            mask = (mask > 0.5).astype(np.uint8) * 255

            # 还原到原始图像尺寸
            mask_orig = cv2.resize(mask, (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST)

            # 用检测框裁剪mask
            bx1 = max(0, int(x1 * scale_x))
            by1 = max(0, int(y1 * scale_y))
            bx2 = min(orig_w, int(x2 * scale_x))
            by2 = min(orig_h, int(y2 * scale_y))
            mask_crop = np.zeros_like(mask_orig)
            mask_crop[by1:by2, bx1:bx2] = mask_orig[by1:by2, bx1:bx2]

            # 过滤面积太小的mask
            if cv2.countNonZero(mask_crop) < 500:
                continue

            # 提取轮廓中心
            contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            max_cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

        return centers

    def detect(self, image):
        """
        检测图像中的镜片

        Args:
            image: BGR格式图像 (numpy.ndarray)

        Returns:
            centers: 镜片中心坐标列表 [(x,y), ...]
                    坐标原点: 图像左上角, x向右为正, y向下为正
                    单位: 像素
            inference_time_ms: 推理耗时(毫秒)
        """
        if image is None or image.size == 0:
            return [], 0

        orig_h, orig_w = image.shape[:2]

        # 预处理
        inp = self._preprocess(image)

        # 拷贝输入到GPU
        np.copyto(self.bindings[self.input_name]['host'], inp)
        cuda.memcpy_htod_async(
            self.bindings[self.input_name]['device'],
            self.bindings[self.input_name]['host'],
            self.stream
        )

        # 绑定张量地址
        for name, buf in self.bindings.items():
            self.context.set_tensor_address(name, int(buf['device']))

        # 推理
        t0 = time.time()
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 拷贝输出到CPU
        outputs = {}
        for name in self.output_names:
            cuda.memcpy_dtoh_async(
                self.bindings[name]['host'],
                self.bindings[name]['device'],
                self.stream
            )
        self.stream.synchronize()

        for name in self.output_names:
            outputs[name] = self.bindings[name]['host'].copy()

        inference_time_ms = (time.time() - t0) * 1000

        # 后处理 - 获取镜片中心坐标
        centers = self._postprocess(outputs, orig_h, orig_w)

        return centers, inference_time_ms

    def detect_with_debug(self, image, save_path=None):
        """
        检测图像中的镜片，并返回带标注的调试图像

        Args:
            image: BGR格式图像
            save_path: 调试图像保存路径 (可选)

        Returns:
            centers: 镜片中心坐标列表
            inference_time_ms: 推理耗时
            debug_image: 带标注的图像
        """
        centers, inference_time_ms = self.detect(image)

        debug_image = image.copy()

        # 绘制检测结果
        for cx, cy in centers:
            cv2.circle(debug_image, (cx, cy), 6, (0, 0, 255), -1)  # 红色中心点
            cv2.putText(debug_image, f"({cx}, {cy})", (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 绘制信息
        cv2.putText(debug_image, f"Lenses: {len(centers)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(debug_image, f"Inference: {inference_time_ms:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if save_path:
            cv2.imwrite(save_path, debug_image)

        return centers, inference_time_ms, debug_image

    def __del__(self):
        """清理CUDA资源"""
        try:
            if hasattr(self, 'stream'):
                self.stream.synchronize()
            # pycuda会自动管理内存释放
        except:
            pass


# ============ 使用示例 ============

def main():
    """使用示例"""
    # 配置
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path(__file__).with_name("best.engine"),
    )
    parser.add_argument("--output", type=Path, default=Path("debug_result.jpg"))
    args = parser.parse_args()
    model_path = args.engine
    image_path = args.image

    # 创建检测器
    detector = LensDetector(model_path, conf_threshold=0.15)

    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return

    print(f"📷 图像尺寸: {image.shape}")

    # 方式1: 只获取坐标
    centers, inference_time = detector.detect(image)
    print(f"✅ 检测到 {len(centers)} 个镜片")
    print(f"⏱️ 推理耗时: {inference_time:.1f}ms")
    for i, (x, y) in enumerate(centers):
        print(f"  镜片 {i+1}: 中心坐标 ({x}, {y})")

    # 方式2: 获取带标注的调试图像
    centers, inference_time, _ = detector.detect_with_debug(
        image,
        save_path=str(args.output)
    )
    print(f"✅ 调试图像已保存: debug_result.jpg")

    # # 批量处理示例
    # image_dir = "images/"
    # for img_file in os.listdir(image_dir):
    #     if img_file.endswith(('.jpg', '.png')):
    #         img = cv2.imread(os.path.join(image_dir, img_file))
    #         centers, _ = detector.detect(img)
    #         print(f"{img_file}: {len(centers)} 个镜片")
    #         for x, y in centers:
    #             print(f"  ({x}, {y})")


if __name__ == '__main__':
    main()
