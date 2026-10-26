# Raspberry Pi Setup

This guide runs the project's Flask dashboard in Docker on a Raspberry Pi. It uses the root-level `Dockerfile` and stores event history in a Docker volume.

## Requirements

- Raspberry Pi 4 or 5 with a 64-bit Raspberry Pi OS installation.
- Internet access during installation and the first image build.
- A USB webcam. The app opens camera index `0` on the Pi.
- SSH access or a terminal on the Pi.

The dashboard uses MediaPipe and OpenCV. Performance depends on the Pi model, camera resolution, and available memory. A USB webcam exposed as `/dev/video0` is the simplest camera setup. CSI camera modules may require additional V4L2 configuration and are not guaranteed to work with the current OpenCV camera capture code.

## 1. Install Docker

Update the Pi and install Docker from the Raspberry Pi OS package repository:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y docker.io
sudo systemctl enable --now docker
```

Allow your user to run Docker without `sudo`, then sign out and back in:

```bash
sudo usermod -aG docker "$USER"
```

Confirm Docker works after signing back in:

```bash
docker version
```

## 2. Copy the project to the Pi

Copy the project folder to the Pi, or clone its repository if it is hosted in Git. For example, from a computer with the project folder:

```bash
scp -r DriverMonitoringSystem pi@<PI_IP_ADDRESS>:/home/pi/
```

Replace `<PI_IP_ADDRESS>` with the Pi's address. Then connect and enter the project directory:

```bash
ssh pi@<PI_IP_ADDRESS>
cd /home/pi/DriverMonitoringSystem
```

The directory should contain `Dockerfile`, `requirements.txt`, `web_app.py`, `wsgi.py`, and the `templates` folder.

## 3. Check the webcam

Connect the webcam and check that Linux exposes it:

```bash
ls -l /dev/video*
```

The default app camera index (`0`) usually corresponds to `/dev/video0`. If the camera appears at a different device or index, the current app configuration may need to be changed.

## 4. Build the image

Build on the Pi so Docker selects the Pi's CPU architecture:

```bash
docker build -t driver-monitor .
```

The image build installs the Python dependencies, including MediaPipe. If the build fails while installing MediaPipe, check that the Pi OS is 64-bit and that a compatible MediaPipe wheel is available for the selected Python version and Pi architecture. The project currently does not pin dependency versions or provide a custom ARM build of MediaPipe.

## 5. Start the dashboard

Create a named volume for persistent event history. The following command passes `/dev/video0` into the container and adds the host device's group ID so the non-root app user can try to read it:

```bash
docker volume create driver-monitor-data
docker run -d \
  --name driver-monitor \
  --restart unless-stopped \
  -p 5000:5000 \
  --device=/dev/video0:/dev/video0 \
  --group-add "$(stat -c '%g' /dev/video0)" \
  -v driver-monitor-data:/data \
  driver-monitor
```

If there is no `/dev/video0`, resolve the camera setup before running the command. If Docker reports a device permission error, check the device's owner and group with `ls -l /dev/video0`, and confirm that the group ID is passed to the container.

## 6. Open the dashboard

Find the Pi's IP address:

```bash
hostname -I
```

On a device on the same trusted network, open:

```text
http://<PI_IP_ADDRESS>:5000
```

Use the dashboard's monitoring controls to start the camera feed. Keep the browser and Pi on the same network. Do not expose this service directly to the public internet; it does not provide user authentication.

## Manage the container

View application output:

```bash
docker logs -f driver-monitor
```

Stop or start the container:

```bash
docker stop driver-monitor
docker start driver-monitor
```

After changing project files, rebuild and recreate the container:

```bash
docker stop driver-monitor
docker rm driver-monitor
docker build -t driver-monitor .
docker run -d \
  --name driver-monitor \
  --restart unless-stopped \
  -p 5000:5000 \
  --device=/dev/video0:/dev/video0 \
  --group-add "$(stat -c '%g' /dev/video0)" \
  -v driver-monitor-data:/data \
  driver-monitor
```

The `driver-monitor-data` volume preserves `/data/driver_monitor.db` when the container is removed or upgraded. To remove the saved event history too, explicitly remove that volume:

```bash
docker volume rm driver-monitor-data
```
