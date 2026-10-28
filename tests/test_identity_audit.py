import gzip,json
from semantic_map.identity_audit import audit

def fixture(tmp_path,reenter_id="A",co_visible=False):
    frames=[[] for _ in range(41)]
    records=[{"associations":[]} for _ in range(41)]
    descriptor=[1.,0.,0.]
    def add(n,ident,detect):
        frames[n].append({"detection_id":detect,"label":"chair","center":[1.,2.,.8],
          "low":[.8,1.8,.5],"high":[1.2,2.2,1.2],"appearance":descriptor})
        records[n]["associations"].append({"detection_id":detect,"object_id":ident})
    add(0,"A","d0")
    add(40,reenter_id,"d1")
    if co_visible:
        add(40,"A","d2")
    with gzip.open(tmp_path/"observation_evidence.json.gz","wt") as f:json.dump(frames,f)
    (tmp_path/"real_frame_associations.json").write_text(json.dumps(records))
    return tmp_path

def test_true_canonical_reentry_remains_single_id(tmp_path):
    report=audit(fixture(tmp_path))
    assert report['consistent_same_id_pairs']==1
    assert not report['suspected_false_split_track_pairs']

def test_non_covisible_duplicate_id_is_only_diagnostic(tmp_path):
    report=audit(fixture(tmp_path,reenter_id="B"))
    assert report['different_id_pairs_without_known_covisibility']==1
    assert report['suspected_false_split_track_pairs'][0]['object_ids']==["A","B"]
    assert report['status']=='PROXY_DIAGNOSTIC_NOT_PHYSICAL_GROUND_TRUTH'

def test_covisible_similar_objects_are_not_auto_merge_candidates(tmp_path):
    report=audit(fixture(tmp_path,reenter_id="B",co_visible=True))
    assert not report['suspected_false_split_track_pairs']
