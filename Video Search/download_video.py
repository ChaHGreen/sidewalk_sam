import sys
from pytube import YouTube
import os

def download_video(url, output_path='videos'):
    try:
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 获取YouTube视频对象
        yt = YouTube(url)
        
        # 获取最高质量的视频流
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        
        # 下载视频
        if video:
            video.download(output_path=output_path)
            print(f"Downloaded video to {os.path.join(output_path, video.default_filename)}")
        else:
            print("No mp4 video available for download.")
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_script.py 'youtube_url'")
        sys.exit(1)
    
    youtube_url = sys.argv[1]
    download_video(youtube_url)

