mkdir -p ~/mosquitto/config ~/mosquitto/data ~/mosquitto/log

cat > ~/mosquitto/config/mosquitto.conf << 'EOF'
listener 1883
allow_anonymous true
persistence true
persistence_location /mosquitto/data/
log_dest file /mosquitto/log/mosquitto.log
log_dest stdout
log_type all
EOF

podman run -d \
  --name mosquitto \
  --restart=always \
  -p 1883:1883 \
  -v ~/mosquitto/config:/mosquitto/config:Z \
  -v ~/mosquitto/data:/mosquitto/data:Z \
  -v ~/mosquitto/log:/mosquitto/log:Z \
  docker.io/eclipse-mosquitto:2.0

sudo firewall-cmd --permanent --add-port=1883/tcp
sudo firewall-cmd --reload
