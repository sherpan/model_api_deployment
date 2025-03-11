 docker build -t myfastapiimage .   # Build the image

 docker run -d --name mycontainer -p 8000:8000 myfastapiimage

 # Wait for the container to be ready
echo "Waiting for the container to be ready..."
sleep 5  # Adjust the sleep duration as needed

 curl -X POST "http://127.0.0.1:8000/inference/" -H "accept: application/json" -H "Content-Type: application/json" -d @sample_input.json -o verify_test_inference.json