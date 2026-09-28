# [우리 파일] Windows 전용 bootstrap 스텁의 Linux no-op 대체.
# vendor 스크립트들이 `import bootstrap`을 하므로 존재만 하면 된다.
# (원본 bootstrap.py는 sys.path 조작 + Linux 의존성 스텁이었음 — 우리 환경은 불필요)
