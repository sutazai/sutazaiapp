#!/bin/bash

echo "Testing Grafana connection..."

# Test health endpoint (no auth required)
echo "1. Testing health endpoint:"
curl -s "http://localhost:10050/api/health" | jq '.'

echo -e "\n2. Testing with stored password:"
GRAFANA_PASSWORD=$(cat /opt/sutazaiapp/secrets/grafana_password.txt 2>/dev/null || echo "password_file_not_found")
echo "Password length: ${#GRAFANA_PASSWORD}"

echo -e "\n3. Testing authentication:"
response=$(curl -s -w "%{http_code}" -u "admin:${GRAFANA_PASSWORD}" "http://localhost:10050/api/user" -o /tmp/grafana_test.json)
echo "HTTP Status: $response"
if [[ "$response" == "200" ]]; then
    echo "Authentication successful!"
    cat /tmp/grafana_test.json | jq '.'
else
    echo "Authentication failed. Response:"
    cat /tmp/grafana_test.json 2>/dev/null || echo "No response body"
fi

echo -e "\n4. Testing with default password:"
response2=$(curl -s -w "%{http_code}" -u "admin:admin" "http://localhost:10050/api/user" -o /tmp/grafana_test2.json)
echo "HTTP Status with 'admin:admin': $response2"
if [[ "$response2" == "200" ]]; then
    echo "Default credentials work!"
    cat /tmp/grafana_test2.json | jq '.'
else
    echo "Default credentials failed."
fi

echo -e "\n5. Checking Grafana logs for clues:"
docker logs sutazai-grafana 2>&1 | tail -10 || echo "Could not get Grafana logs"

rm -f /tmp/grafana_test.json /tmp/grafana_test2.json