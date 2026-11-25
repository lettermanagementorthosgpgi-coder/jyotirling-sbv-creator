"""SBV-first Shorts research and JSON manifest builder."""
import json, os, re
from pathlib import Path
from .ai_providers import generate

TS = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*,\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*$")
PROVIDERS = ("openai", "gemini", "cloudflare")

def _sec(h,m,s,ms):
    return int(h)*3600 + int(m)*60 + int(s) + int(str(ms).ljust(3,"0")[:3])/1000

def parse_sbv(text):
    lines=text.replace("\ufeff","").replace("\r\n","\n").replace("\r","\n").split("\n")
    cues=[]; i=0
    while i<len(lines):
        if not lines[i].strip(): i+=1; continue
        match=TS.match(lines[i])
        if not match: raise ValueError(f"Invalid SBV timestamp at line {i+1}: {lines[i]!r}")
        v=match.groups(); start=_sec(*v[:4]); end=_sec(*v[4:])
        if end<=start: raise ValueError(f"SBV cue at line {i+1} ends before it starts.")
        i+=1; caption=[]
        while i<len(lines) and lines[i].strip(): caption.append(lines[i]); i+=1
        spoken="\n".join(caption).strip()
        if spoken: cues.append({"start":round(start,3),"end":round(end,3),"text":spoken})
    if not cues: raise ValueError("The SBV file contains no readable caption cues.")
    validate_cues(cues)
    return cues

def validate_cues(cues):
    for n,c in enumerate(cues,1):
        if c["start"]<0 or c["end"]<=c["start"]: raise ValueError(f"Invalid cue timing at cue {n}.")
    for n,(a,b) in enumerate(zip(cues,cues[1:]),1):
        if b["start"]<a["start"]: raise ValueError("SBV cues are not in chronological order.")
        if b["start"]<a["end"]-0.05: raise ValueError(f"Overlapping SBV cues near cue {n}.")

def make_batches(cues, window_seconds=300):
    batches=[]; current=[]; window_start=cues[0]["start"]
    for cue in cues:
        if current and cue["start"]-window_start>=window_seconds:
            batches.append(current); current=[]; window_start=cue["start"]
        current.append(cue)
    if current: batches.append(current)
    if sum(map(len,batches))!=len(cues): raise AssertionError("Transcript batching lost SBV cues.")
    return batches

def _json(raw):
    value=raw.strip(); fence=chr(96)*3
    if value.startswith(fence): value=value[len(fence):].lstrip()
    if value.lower().startswith("json"): value=value[4:].lstrip()
    if value.endswith(fence): value=value[:-len(fence)].rstrip()
    try: result=json.loads(value)
    except json.JSONDecodeError:
        a,b=value.find("{"),value.rfind("}")
        if a<0 or b<=a: raise ValueError("AI response did not contain JSON.")
        result=json.loads(value[a:b+1])
    if not isinstance(result,dict): raise ValueError("AI response must be a JSON object.")
    return result

def _prompt(batch, number, total, minimum, maximum):
    return f"""You independently select YouTube Shorts from a Hindi spiritual discourse.
Transcript batch {number}/{total}. Analyze every supplied timestamped SBV cue.
SBV is authoritative for spoken words. Never invent, paraphrase, rewrite, or add speech.
Find up to 4 self-contained clips of {minimum}-{maximum} seconds. Begin/end at natural thought boundaries;
include setup and conclusion. Prefer compelling, emotional, practical, explanatory or spiritually meaningful passages.
Reject greetings, dead air, weak introductions, incomplete questions, mid-sentence cuts and context-dependent fragments.
Do not force a target count. Use only supplied timestamps.
Return ONLY JSON: {{"candidates":[{{"start_seconds":12.3,"end_seconds":39.8,"title":"Clickable Hindi title","description":"Accurate concise description","quality_score":9.2,"reason":"Why complete and compelling"}}]}}
Score 0-10 for completeness, hook, value, context independence and faithful meaning.
Do not output subtitles; Python will reconstruct them verbatim from SBV.
Transcript cues: {json.dumps(batch,ensure_ascii=False,separators=(",",":"))}"""

def _candidate(item,cues,provider,minimum,maximum):
    try: start=round(float(item["start_seconds"]),3); end=round(float(item["end_seconds"]),3); score=float(item.get("quality_score",0))
    except (KeyError,TypeError,ValueError): return None
    if end-start<minimum or end-start>maximum or score<8: return None
    if not any(abs(c["start"]-start)<=0.25 for c in cues): return None
    if not any(abs(c["end"]-end)<=0.25 for c in cues): return None
    title=str(item.get("title","")).strip(); desc=str(item.get("description","")).strip(); reason=str(item.get("reason","")).strip()
    if not title or not desc or not reason: return None
    return {"start_seconds":start,"end_seconds":end,"title":title[:100],"description":desc[:1200],
            "quality_score":min(10,score),"reason":reason[:1000],"provider":provider}

def _subtitles(cues,start,end):
    out=[]
    for c in cues:
        left=max(start,c["start"]); right=min(end,c["end"])
        if right<=left or not c["text"].strip(): continue
        out.append({"start_seconds":round(max(0,left-start),3),"end_seconds":round(max(0,right-start),3),"text":c["text"]})
    return out

def _select(candidates,count):
    groups={}
    for item in candidates: groups.setdefault((round(item["start_seconds"],1),round(item["end_seconds"],1)),[]).append(item)
    merged=[]
    for items in groups.values():
        best=max(items,key=lambda x:x["quality_score"]); reviewers=sorted({x["provider"] for x in items})
        merged.append({**best,"reviewers":reviewers,"quality_score":round(min(10,best["quality_score"]+0.25*(len(reviewers)-1)),2)})
    merged.sort(key=lambda x:(len(x["reviewers"]),x["quality_score"]),reverse=True)
    chosen=[]
    for item in merged:
        s,e=item["start_seconds"],item["end_seconds"]
        if any(s<old["end_seconds"] and e>old["start_seconds"] for old in chosen): continue
        chosen.append(item)
        if len(chosen)>=count: break
    return sorted(chosen,key=lambda x:x["start_seconds"])

def create_manifest(sbv_path,output_path,count=10,minimum=10,maximum=60):
    path=Path(sbv_path); cues=parse_sbv(path.read_text(encoding="utf-8-sig"))
    batches=make_batches(cues,int(os.getenv("SBV_BATCH_SECONDS","600")))
    available=[]
    if os.getenv("OPENAI_API_KEY"): available.append("openai")
    if os.getenv("GEMINI_API_KEY"): available.append("gemini")
    if os.getenv("CLOUDFLARE_API_TOKEN") and os.getenv("CLOUDFLARE_ACCOUNT_ID"): available.append("cloudflare")
    missing=sorted(set(PROVIDERS)-set(available))
    if missing: raise RuntimeError("Three independent AI reviewers required; missing: "+", ".join(missing))
    candidates=[]; failures=[]
    for n,batch in enumerate(batches,1):
        print(f"Reviewing complete transcript batch {n}/{len(batches)} ({len(batch)} cues).",flush=True)
        for provider in available:
            try:
                raw,model,_usage=generate(provider,_prompt(batch,n,len(batches),minimum,maximum))
                items=_json(raw).get("candidates",[])
                if not isinstance(items,list): raise ValueError("AI candidates must be a list.")
                accepted=0
                for item in items:
                    if isinstance(item,dict):
                        found=_candidate(item,batch,provider,minimum,maximum)
                        if found: found["model"]=model; candidates.append(found); accepted+=1
                print(f"{provider} ({model}): {accepted} candidate(s).",flush=True)
            except Exception as exc:
                failures.append(f"{provider} batch {n}: {type(exc).__name__}: {exc}")
                print("WARNING: "+failures[-1],flush=True)
    if not candidates: raise RuntimeError("No valid candidates. "+" | ".join(failures[:6]))
    selected=_select(candidates,count)
    if not selected: raise RuntimeError("No non-overlapping candidates passed validation.")
    shorts=[]
    for item in selected:
        subs=_subtitles(cues,item["start_seconds"],item["end_seconds"])
        if not subs: raise RuntimeError("Selected clip has no SBV subtitles; refusing empty captions.")
        shorts.append({"start_seconds":item["start_seconds"],"end_seconds":item["end_seconds"],
                       "title":item["title"],"description":item["description"],"subtitles":subs})
    manifest={"shorts":shorts}
    Path(output_path).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report={"sbv_file":path.name,"cue_count":len(cues),"batch_count":len(batches),"providers":available,
            "candidate_count":len(candidates),"selected_count":len(shorts),
            "selected":[{k:x[k] for k in ("start_seconds","end_seconds","title","quality_score","reason","reviewers")} for x in selected],
            "warnings":failures}
    Path("sbv_review_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Manifest ready: {len(shorts)} Shorts from all {len(cues)} SBV cues.",flush=True)
    return manifest
