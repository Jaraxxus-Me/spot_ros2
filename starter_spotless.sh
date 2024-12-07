echo "Checking WIFI Connection..."
echo ping 192.168.80.3
python3 -m bosdyn.client 192.168.80.3 id

export BOSDYN_CLIENT_USERNAME=admin
export BOSDYN_CLIENT_PASSWORD=8bjvkcjtghki

python3 src/spot_ros2/hello_spot.py 192.168.80.3