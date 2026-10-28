"""Fail-closed release gate: no negative or missing result can be called accepted."""
import argparse
import json
from pathlib import Path

ALLOWED={"PASS","FAIL","NOT_TESTED","IMPLEMENTED_UNVALIDATED"}

def evaluate(doc):
    stages=doc["required_stages"]
    if not stages:raise ValueError("Empty acceptance register")
    for name,v in stages.items():
        if v.get("status") not in ALLOWED:raise ValueError("Unknown acceptance state "+str(name))
    failures={k:v["status"] for k,v in stages.items() if v["status"]!="PASS"}
    return len(failures)==0, failures

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--file",default="config/acceptance.json")
    p.add_argument("--assert-complete",action="store_true")
    a=p.parse_args()
    success,not_passed=evaluate(json.loads(Path(a.file).read_text()))
    print(json.dumps({"final_project_accepted":success,"blocking_stages":not_passed},ensure_ascii=False,indent=2))
    if a.assert_complete and not success:raise SystemExit(2)

if __name__=="__main__":main()
