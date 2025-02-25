#!/bin/bash
sutazai-cli code \
  --generate "optimized fibonacci function" \
  --language $(sutazai-cli suggest-language mathematical) \
  --output fib.$(sutazai-cli suggest-extension mathematical) 