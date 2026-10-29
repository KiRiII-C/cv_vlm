from semantic_map.hypersim_full_sequence_eval import gap_stats,_evolving_scores
from semantic_map.hypersim_full_sequences import discover


def _timeline(visible,assigned):
    frames=[];tl=[]
    for t,(vis,assignment) in enumerate(zip(visible,assigned)):
        gt=[7] if assignment else []
        frames.append(dict(frame=t,target_visible=[7] if vis else [],
                           all_gt_visible=[7] if vis else [],
                           physical_gt=gt,observations=[{}] if gt else []))
        tl.append(dict(frame=t,uuids=[assignment] if assignment else [],statuses=[]))
    return dict(scene="synthetic",frames=frames),tl


def test_five_frame_out_of_view_reentry_same_physical_uuid():
    cache,tl=_timeline([True,True,False,False,False,False,False,True,True],
                       ["u1","u1",None,None,None,None,None,"u1","u1"])
    report=gap_stats(cache,tl)["gaps"]
    row=report["gt_out_of_view_reentry_at_least_5_missing_frames"]
    assert row["eligible_physical_reappearance_pairs"]==1
    assert row["uuid_retained"]==1 and row["uuid_switched"]==0
    assert report["per_physical_object"]["observed_uuid_switch_transitions"]==0


def test_physical_uuid_switch_on_real_return():
    cache,tl=_timeline([True,True,False,False,False,False,False,True,True],
                       ["u1","u1",None,None,None,None,None,"u2","u2"])
    report=gap_stats(cache,tl)["gaps"]
    row=report["gt_out_of_view_reentry_at_least_5_missing_frames"]
    assert row["uuid_switched"]==1 and row["uuid_retained"]==0
    assert report["per_physical_object"]["gt_objects_with_two_or_more_assigned_uuids"]==1
    assert report["per_physical_object"]["observed_uuid_switch_transitions"]==1


def test_detector_dropout_not_falsely_called_out_of_view():
    cache,tl=_timeline([True]*9,["u1",None,None,None,None,None,None,None,"u2"])
    gaps=gap_stats(cache,tl)["gaps"]
    assert gaps["detector_gaps_at_least_5_missing_frames"]["uuid_switched"]==1
    assert gaps["gt_out_of_view_reentry_at_least_5_missing_frames"]["eligible_physical_reappearance_pairs"]==0


def test_100_frame_evolution_does_not_skip_frames():
    cache,tl=_timeline([True]*100,["u"]*100)
    windows=_evolving_scores(cache,tl)
    assert len(windows)==4
    assert [x["frames"] for x in windows]==[25,25,25,25]
    assert all(x["local_window"]["IDF1"]==1 for x in windows)
    assert windows[-1]["cumulative"]["gt_observations"]==100


def test_inventory_rejects_missing_geometry():
    class Archive:
        def namelist(self):
            return [f"ai_001_001/images/scene_cam_00_{source}/frame.{i:04d}.{kind}"
                    for i in range(3)
                    for source,kind in [
                        ("final_preview","tonemap.jpg"),
                        ("geometry_hdf5","semantic_instance.hdf5"),
                        ("geometry_hdf5","semantic.hdf5"),
                        ("geometry_hdf5","position.hdf5")
                    ] if not(i==1 and kind=="position.hdf5")]
    info=discover(Archive(),"ai_001_001")["cam_00"]
    assert info["union_count"]==3 and info["complete_count"]==2
    assert info["missing_modalities"]==[1]
