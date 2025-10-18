#!/bin/bash
API_KEY="9L9XbOYuViAriFEAqy"
SECRET_KEY="VBcajUK3rIqj8tUJpexfy4H0AOqjuf7l909I"
TIMESTAMP=$(date +%s000)
RECV_WINDOW="5000"
QUERY_STRING="category=linear&settleCoin=USDT"
SIGN_PAYLOAD="${TIMESTAMP}${API_KEY}${RECV_WINDOW}${QUERY_STRING}"
SIGNATURE=$(echo -n "${SIGN_PAYLOAD}" | openssl dgst -sha256 -hmac "${SECRET_KEY}" | awk '{print $2}')
curl -s "https://api-demo.bybit.com/v5/position/list?${QUERY_STRING}" -H "X-BAPI-API-KEY: ${API_KEY}" -H "X-BAPI-TIMESTAMP: ${TIMESTAMP}" -H "X-BAPI-SIGN: ${SIGNATURE}" -H "X-BAPI-RECV-WINDOW: ${RECV_WINDOW}"
