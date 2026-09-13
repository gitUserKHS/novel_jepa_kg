# 캡스톤디자인II 중간 점검 보고

LMS 제출 마감: **2026년 9월 14일(월) 오후 11시**

| 파일 | 용도 |
|---|---|
| `midterm_report_2026-09-14.html` | 원본. 여기서 고친다 |
| `midterm_report_2026-09-14.pdf` | 제출용 (A4 6쪽). HTML 에서 다시 뽑는다 |

## 제출 전에 채울 것

HTML 의 `[기입 필요]` 두 곳 — **팀명**, **팀원**. 빨간 글씨라 찾기 쉽다.

## PDF 다시 뽑기

브라우저로 HTML 을 열고 `Ctrl+P` → 대상 "PDF로 저장" → 머리글/바닥글 끄기.

명령줄로 뽑으려면 (Windows, Edge):

```powershell
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless --no-pdf-header-footer `
  --print-to-pdf="$PWD\midterm_report_2026-09-14.pdf" "file:///$PWD/midterm_report_2026-09-14.html"
```

## 수치의 출처

보고서의 숫자는 추정이 아니라 저장소에서 센 값이다. 다시 확인하려면:

- 커밋 수 (캡스톤II 구간): `git log --since=2026-07-01 --oneline | wc -l`
- 작업 수: `TASKS.md` 의 Phase 7~9 체크 항목
- 테스트 수: `.venv\Scripts\python.exe -m unittest discover -s tests`
- JEPA 보정 수치(1.6 절): 커밋 `5650311` 직전 README 의 plausibility calibration 표
- 성능 실측(1.7 절): `README.md` 의 "속도" 절, `TASKS.md` Phase 9

완성도 90% 와 월별 누적 진척(40/75/90%)은 팀의 판단이 들어간 값이다. 조정하려면 2 절의 표만 바꾸면 된다.
