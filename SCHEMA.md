# Narrative Transition JSON Schema

Each sample is a scene transition pair used to train the JEPA-inspired latent predictor.

## Required fields

- `world`: world setting object
- `characters`: character list
- `scene_t`: current scene
- `scene_t_plus_1`: next scene

## Scene fields

- `summary`: concise event summary
- `emotion`: emotional state or transition
- `conflict`: central tension
- `state`: list of important state tags
- `plot_function`: narrative role such as setup, escalation, reveal, reversal, aftermath

## Example

```json
{
  "world": {
    "genre": "SF 판타지",
    "premise": "별의 기억이 물리 현상으로 발현되는 세계",
    "rules": ["기억의 공명은 감정 변화와 연결된다"]
  },
  "characters": [
    {
      "name": "주인공A",
      "role": "탐색자",
      "goal": "사라진 동생의 흔적을 찾는다",
      "fear": "기억을 잃는 것",
      "relationship": "동료B를 신뢰하지만 완전히 의지하지 못함"
    }
  ],
  "scene_t": {
    "summary": "주인공A가 폐허 지하철역에서 이상한 신호를 발견한다.",
    "emotion": "불안",
    "conflict": "도망칠지 조사할지 고민한다.",
    "state": ["밤", "폐허", "약한 신호", "체력 저하"],
    "plot_function": "사건 발단"
  },
  "scene_t_plus_1": {
    "summary": "신호가 동생의 목소리와 닮았다는 사실을 깨닫고 안쪽으로 들어간다.",
    "emotion": "불안 → 희망",
    "conflict": "위험을 알면서도 진입한다.",
    "state": ["지하 통로 진입", "신호 강화", "새 목표 발생"],
    "plot_function": "목표 강화"
  }
}
```
