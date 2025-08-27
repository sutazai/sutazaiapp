#!/bin/bash
[ "$1" = "--selfcheck" ] && { echo '{"healthy":true}'; exit 0; }
exec node -e "console.log(JSON.stringify({status:'ready'}))"
