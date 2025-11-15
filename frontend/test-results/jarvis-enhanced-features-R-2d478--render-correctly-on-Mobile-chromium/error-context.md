# Page snapshot

```yaml
- generic [ref=e1]:
  - banner [ref=e2]:
    - generic [ref=e4]:
      - button "Deploy" [ref=e6] [cursor=pointer]:
        - generic [ref=e8]: Deploy
      - button [ref=e10] [cursor=pointer]:
        - img [ref=e11]
  - generic [ref=e14]:
    - button [ref=e17] [cursor=pointer]:
      - img [ref=e18]
    - generic [ref=e24]:
      - button [ref=e27] [cursor=pointer]:
        - img [ref=e28]
      - generic [ref=e35]:
        - heading "🎮 Control Panel" [level=2] [ref=e39]
        - heading "🤖 AI Model" [level=3] [ref=e43]
    - generic [ref=e52]:
      - generic [ref=e57]:
        - heading "J.A.R.V.I.S" [level=1] [ref=e66]: J.A.R.V.I.S
        - paragraph [ref=e71]: Just A Rather Very Intelligent System
      - generic [ref=e72]: "Backend: Connected"
      - alert [ref=e75]:
        - generic [ref=e77]:
          - generic [ref=e78]: "AttributeError: st.session_state has no attribute \"current_model\". Did you forget to initialize it? More info: https://docs.streamlit.io/develop/concepts/architecture/session-state#initialization"
          - generic [ref=e79]: "Traceback:"
          - code [ref=e82]:
            - generic [ref=e83]: File "/app/app.py", line 452, in <module> if st.session_state.current_model in st.session_state.available_models else 0, ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            - generic [ref=e84]: File "/usr/local/lib/python3.11/site-packages/streamlit/runtime/state/session_state_proxy.py", line 131, in __getattr__ raise AttributeError(_missing_attr_error_message(key))
```