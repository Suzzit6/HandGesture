from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL


def setVolume(volume):
  
# Get system volume control
 devices = AudioUtilities.GetSpeakers()
 interface = devices.Activate(
     IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
 volume_controller = interface.QueryInterface(IAudioEndpointVolume)
 
 volume_controller.SetMasterVolumeLevelScalar(volume, None)
 print(f"Volume set to",volume)
