# to test the deployment of the model, replace the url with your own IP address and port number

curl http://120.77.8.29:12345/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "./Mistral-7B-Instruct-v0.2",
        "prompt": "Once a upon time,",
        "max_tokens": 8,
        "stream": false,
        "top_p": 1.0
    }'