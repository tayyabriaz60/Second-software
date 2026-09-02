# Client share — what to send (checklist)

## Attach / upload
1. `CLIENT_UPDATE_BRIEF.md` (this folder) — status + improvements  
2. Latest demo MP4 from RunPod (e.g. `output_mot_v6_render.mp4` or `output_mot_v6_10min.mp4`)  
3. Matching `player_id_list_*_mot_sota_v6_*.json`

## Short message you can paste (email / WhatsApp)

Subject: Soccer tracking update — detection ready, identity in progress

Hi,

Sharing a progress update on the panoramic camera pipeline.

What improved since the last review:
- Player detection remains solid (RF-DETR on GPU).
- The big “ID explosion” bug (hundreds of fake 100xxx IDs) is largely fixed (428 → 34 collision IDs on the benchmark run).
- Tracking now renders correctly on video (earlier runs could show a blank output when pitch keypoints failed).
- Pass-2 render is faster (no redundant appearance model on every frame).

Attached:
1) Short status brief  
2) Demo video with player boxes / IDs  
3) ID list JSON for the same run  

Honest status: detection and reviewable tracking are ready. Full “one stable ID per player for 10 minutes” (so distance stats are trustworthy) is still the active workstream — that is next, before ball tracking.

Happy to walk through the video on a call.

Thanks
