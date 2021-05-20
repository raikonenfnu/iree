Those headers come from ROCM SDK.

Currently updates are not supported by ROCm, so we need to uninstall and reinstall ROCm if we want to update
To update, install ROCM SDK locally:
```
sudo apt autoremove rocm-opencl rocm-dkms rocm-dev rocm-utils && sudo reboot
sudo apt-get install rocm-dkms
```

Copy ROCM AMDGN ockl.bc nad ocml.bc:
```
cp -RL /opt/rocm/amdgcn/ockl.bc ./amdgcn/.
cp -RL /opt/rocm/include/ocml.bc ./amdgcn/.
cp /opt/rocm/.info/version version.txt
```