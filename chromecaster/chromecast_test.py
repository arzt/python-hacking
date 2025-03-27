import pychromecast

if __name__ == "__main__":
    chromecasts, browser = pychromecast.get_listed_chromecasts(friendly_names=["Diele"])
    cast = chromecasts[0]
    cast.wait()

    import socket
    host = socket.gethostbyname(socket.gethostname())

    mc = cast.media_controller
    #mc.play_media('http://dell-ix:8000/hdcpchbtkk', 'video/mp4')
    mc.play_media('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4', 'video/mp4')
    mc.block_until_active()
    status = mc.status
    print(status)
