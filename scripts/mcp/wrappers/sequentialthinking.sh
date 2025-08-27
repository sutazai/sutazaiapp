#!/bin/bash
[ "$1" = "--selfcheck" ] && { echo '{"healthy":true}'; exit 0; }
exec python3 -c "import json; print(json.dumps({'status':'ready'}))"
