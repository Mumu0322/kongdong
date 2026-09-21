from types import SimpleNamespace
import numpy as np
import pytest

from aubo_workbench.coarse_surface import ring_coverage, save_surface_diagnostic
from aubo_workbench.coarse_recovery import queue_singleton_recaptures
from aubo_workbench.fitting import fit_plane_model
from aubo_workbench.hole_localization_vision import hole_camera_point, COARSE_SURFACE_SELECTION_POLICY
from aubo_workbench.hole_localization_models import PlaneEstimate, Observation, TwoStageConfig
from aubo_workbench.group_pose_workflow import _pose_correction_motion_gate
import run_yolo_eye_in_hand_optimized as runner


def scene(tilt_deg=18, bottom=False, missing=False):
    intr = SimpleNamespace(width=240, height=240, fx=600., fy=600., cx=120., cy=120., distortion=())
    yy, xx = np.mgrid[:240, :240]
    xray = (xx-120)/600.
    z = 340/(1-np.tan(np.deg2rad(tilt_deg))*xray)
    # A deep hole occupies the inner radial band, leaving a thin external surface.
    if bottom:
        z[np.hypot(xx-120, yy-120) < 60*1.23] += 40
    xyz = np.stack((xray*z, (yy-120)/600.*z, z), axis=-1)
    if missing:
        xyz[xx > 120] = np.nan
    return xyz, intr


def measure(**kwargs):
    xyz, intr = scene(**kwargs)
    return hole_camera_point((120, 120), xyz, intr, 60,
                             surface_selection_policy=COARSE_SURFACE_SELECTION_POLICY,
                             include_points=True)


@pytest.mark.parametrize("tilt", [0, 10, 18, 25])
def test_tilted_annulus_recovers_far_side_without_depth_cut(tilt):
    point, info = measure(tilt_deg=tilt)
    assert abs(point[2]-340) < 0.02
    assert info['ring_coverage_ratio'] == 1
    assert info['ring_max_gap_deg'] == 0
    assert info['surface_points_selected']/info['ring_points_raw'] > .99
    p = info['points_camera_mm']; n = np.array(info['plane_normal_camera'])
    anchor = np.array(info['local_plane_point_camera_mm'])
    assert np.max(np.abs((p-anchor)@n)) < .001


def test_tilted_surface_does_not_grow_into_dominant_hole_bottom():
    point, info = measure(bottom=True)
    assert abs(point[2]-340) < .05
    assert info['ring_coverage_ratio'] == 1
    raw = info['raw_points_camera_mm']; pixels = info['raw_pixels']
    mask = info['surface_selected_mask']
    assert not np.any(np.linalg.norm(pixels[mask]-[120,120],axis=1) < 60*1.23)
    assert len(raw) > len(info['points_camera_mm']) * 2


def test_missing_half_ring_is_not_invented_and_is_rejected_for_formal_fusion(tmp_path):
    _, info = measure(missing=True)
    plane = runner._plane_estimate_from_info(info, 'test')
    assert info['ring_coverage_ratio'] < .6
    observations = [Observation('batch_coarse', i, np.array([120.,120.]), plane=plane) for i in range(10)]
    with pytest.raises(RuntimeError, match='环带覆盖不足'):
        runner._fuse_coarse(observations, TwoStageConfig(), min_valid_frames=8)
    path = tmp_path/'surface.npz'
    save_surface_diagnostic(path, plane, hole_id=21, frame_index=9)
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_allclose(z['raw_points_camera_mm'][z['selected_mask']], info['points_camera_mm'])
        assert z['hole_id'] == 21


def test_fused_center_excludes_low_coverage_frames():
    observations=[]
    for i in range(10):
        plane=PlaneEstimate(np.array([100. if i==9 else 0.,0.,340.]), np.array([0.,0.,-1.]),
                            0.1,100,ring_coverage_ratio=.5 if i==9 else 1.,ring_max_gap_deg=180 if i==9 else 0)
        observations.append(Observation('batch_coarse',i,np.array([0.,0.]),plane=plane))
    s=runner._fuse_coarse(observations,TwoStageConfig(),min_valid_frames=8)
    assert s['valid_frames']==9 and s['ring_quality_rejected_frames']==1
    assert s['accepted_frame_indices']==list(range(9))
    np.testing.assert_allclose(s['plane_point_camera_mm'],[0,0,340])


def test_wraparound_gap_and_single_pixels_do_not_count_as_supported_sectors():
    angles=np.deg2rad(np.arange(85,276,1))
    pixels=np.column_stack([100*np.cos(angles),100*np.sin(angles)])
    quality=ring_coverage(np.vstack([pixels,[100,0]]),(0,0))
    assert quality['max_gap_deg']>=150


def test_plane_anchor_and_rmse_describe_same_inlier_plane():
    rng=np.random.default_rng(9)
    xy=rng.uniform(-20,30,(1000,2))
    z=340+.3*xy[:,0]-.2*xy[:,1]+rng.normal(0,.1,1000)
    z[:15]+=30
    points=np.column_stack([xy,z])
    normal,anchor,rmse,mask=fit_plane_model(points)
    assert abs(rmse-np.sqrt(np.mean(((points[mask]-anchor)@normal)**2)))<1e-10
    assert abs(normal@anchor/normal[2]-340)<.05


def test_motion_roundoff_is_accepted_but_real_excess_is_rejected():
    cfg=TwoStageConfig()
    assert _pose_correction_motion_gate(3.265,2.0000000000006293,cfg)==(True,None)
    assert not _pose_correction_motion_gate(3.265,2.001,cfg)[0]
    assert not _pose_correction_motion_gate(5.001,1,cfg)[0]


def test_retry_only_failed_holes_once_and_keeps_original_seed():
    group=[dict(hole_id=i,initial_center_base_mm=[i,0,0],coarse_pose_refined_center_base_mm=[999,0,0]) for i in [1,2,3]]
    groups=[group];attempted=set()
    assert queue_singleton_recaptures(groups,group,[2],attempted,1)==[2]
    assert groups[-1][0]['initial_center_base_mm']==[2,0,0]
    assert 'coarse_pose_refined_center_base_mm' not in groups[-1][0]
    assert queue_singleton_recaptures(groups,group,[2],attempted,1)==[]
    assert queue_singleton_recaptures(groups,groups[-1],[2],attempted,2)==[]
    assert len(groups)==2


@pytest.mark.parametrize('retry_succeeds', [True, False])
def test_shared_stage_recaptures_only_bad_hole_and_preserves_good_coordinates(tmp_path, monkeypatch, retry_succeeds):
    from aubo_workbench import sequential_workflow_shared as stage
    stage.install_runtime(vars(runner))
    holes=[dict(hole_id=i, initial_center_base_mm=[i,0,0]) for i in [1,2]]
    unused=('model runtime pose_session motion_session initial_intrinsics results order_ids '
            'initial_pointcloud_reused_holes cache_gates coarse_cache_dir persistent_cache_dir '
            'shared_cache_results shared_cache_failed_ids invalidated_cache_ids batch_fine_results batch_fine_plan').split()
    ctx=SimpleNamespace(**{key:None for key in unused})
    ctx.args=SimpleNamespace(map_build_coarse_only=True,confidence=.5)
    ctx.cfg=TwoStageConfig(coarse_settle_delay_s=0)
    ctx.handeye=SimpleNamespace(T_tcp_rgb_camera=np.eye(4))
    ctx.run_dir=tmp_path;ctx.report={'stages':{}};ctx.timing=runner.TimingRecorder();ctx.rows=[]
    ctx.current_tcp=np.eye(4);ctx.initial_holes=holes;ctx.fixed_rz_rad=0
    ctx.all_selected_two_capture_mode=False
    ctx.cache_enabled=ctx.persistent_enabled=False
    ctx.cache_entries={};ctx.cache_sources={};ctx.cache_source_ids={};ctx.persistent_entries={};ctx.cache_built_ids=set()
    ctx.batch_coarse_results={};ctx.batch_coarse_for_cache=True
    ctx.ensure_rgbd_pipeline=lambda:(None,None,None)
    monkeypatch.setattr(stage,'_split_batch_localization_groups',lambda *a,**k:([holes],[],{}))
    for name in ['_save_grouping_plan_visualization','_save_group_capture_visualization']:
        monkeypatch.setattr(stage,name,lambda *a,**k:None)
    def plan(group,*args):
        return np.eye(4),dict(target_tcp_pose_m_rad=[0]*6,projected_holes_px={h['hole_id']:np.array([100,100]) for h in group},
                             group_bbox_px=[100]*4,group_center_px=np.array([100,100]))
    monkeypatch.setattr(stage,'_plan_batch_coarse_group_pose',plan)
    moves=[]
    def move(index,count,current,target,*args,**kwargs):
        moves.append(index);return target
    monkeypatch.setattr(stage,'_move_to_shared_coarse_pose',move)
    refined=[]
    def refine(index,count,group,current,target,geometry,*args):
        refined.append([h.get('coarse_quality_retry',False) for h in group])
        return dict(group=group,current_tcp=current,target=target,geometry=geometry,report={'accepted':True})
    monkeypatch.setattr(stage,'_refine_shared_coarse_group_pose',refine)
    captures=[]
    def capture(group,*args,**kwargs):
        ids=[h['hole_id'] for h in group];captures.append(ids)
        if len(captures)==1:
            return {1:dict(success=True,center_base_mm=[11,12,13]),2:dict(success=False,error='ring quality'),'_batch_metadata':{}}
        return {2:dict(success=retry_succeeds,center_base_mm=[21,22,23] if retry_succeeds else None),'_batch_metadata':{}}
    monkeypatch.setattr(stage,'_batch_coarse_localization_at_340mm',capture)
    stage.run_shared_coarse_stage(ctx)
    assert captures==[[1,2],[2]] and moves==[1,2]
    assert refined==[[False,False],[True]]
    assert ctx.batch_coarse_results[1]['center_base_mm']==[11,12,13]
    assert ctx.batch_coarse_results[2]['success']==retry_succeeds
    plan_report=ctx.report['stages']['batch_coarse_plan']
    assert plan_report['groups'][0]['results'][1]['center_base_mm']==[11,12,13]
    assert plan_report['quality_retry_holes']==[2]
    if not retry_succeeds:
        from aubo_workbench import sequential_hole_execution as execution
        execution.install_runtime(vars(runner))
        ctx.results=[];ctx.shared_cache_results={};ctx.batch_fine_results={}
        holes[1].update(initial_detection={},initial_plane_normal_base=[0,0,1])
        monkeypatch.setattr(execution,'_append_deferred_hole_result',
                            lambda hole,result,results,*args:results.append(result), raising=False)
        def forbidden(*args,**kwargs):
            pytest.fail('exhausted retry must not navigate or start another capture')
        monkeypatch.setattr(execution,'_plan_hole_tcp_pose_fixed_rz',forbidden)
        ctx.ensure_rgbd_pipeline=forbidden
        execution._process_one_hole(ctx,2,holes[1])
        assert ctx.results[0]['status']=='deferred_coarse_quality'


@pytest.mark.parametrize('recovered', [True,False])
@pytest.mark.parametrize('reason', [
    'not_all_holes_have_stable_geometry',
    'pose_correction_exceeds_motion_gate',
    'total_pose_correction_exceeds_motion_gate',
])
def test_map_requires_successful_replacement_of_rejected_geometry_group(recovered, reason):
    from aubo_workbench.hole_map import build_hole_map_payload
    from test_hole_map import _completed_result
    groups=[dict(group_index=1,hole_ids=[1],quality_retry_holes=[1],
                 accepted_holes=[],
                 pose_refinement=dict(accepted=False,reason=reason)),
            dict(group_index=2,hole_ids=[1],coarse_quality_retry=True,coarse_quality_retry_parent_group=1,
                 accepted_holes=[1] if recovered else [],pose_refinement=dict(accepted=True))]
    report=dict(configuration=dict(batch_coarse_localization=True),map_build_localization_mode='coarse_only',
                stages={'batch_coarse_results':{'groups':groups}},final_result={'holes':[_completed_result(1)]})
    kwargs=dict(map_id='test',source_run_dir='test',handeye_path=None,camera_identity=None)
    if recovered:
        payload=build_hole_map_payload(report,**kwargs)
        assert payload['holes']
    else:
        with pytest.raises(ValueError,match='未收敛'):
            build_hole_map_payload(report,**kwargs)
