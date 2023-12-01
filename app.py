import streamlit as st
import os
from helpers.add_subtitles_to_video import add_subtitles
from helpers.whisper_transcribe import transcribe
from helpers.save_video import save_uploaded_file
import datetime


def main():
    st.title("视频转录和字幕")
    now = datetime.datetime.now()
    uploaded_file = st.file_uploader("上传您的视频", type=["mp4"])

    if uploaded_file is not None:
        video_path = save_uploaded_file(uploaded_file)
        st.session_state['video_path'] = video_path

    if 'video_path' in st.session_state and st.button("生成转录"):
        st.session_state['srt_file_path'] = transcribe(st.session_state['video_path'])
        with open(st.session_state['srt_file_path'], 'r') as file:
            st.session_state['transcription_text'] = file.read()

    if 'transcription_text' in st.session_state:
        confirmed_transcription = st.text_area("如有需要，请编辑字幕：", st.session_state['transcription_text'], height=500)
        if st.button("保存转录"):
            with open(st.session_state['srt_file_path'], 'w') as file:
                file.write(confirmed_transcription)
            st.session_state['save_clicked'] = True
            st.success("转录已保存。")

    if st.session_state.get('save_clicked', False):
        if st.button("添加字幕"):

            add_subtitles(st.session_state['video_path'], st.session_state['srt_file_path'], f"video_with_subtitles{now}.mp4")
            st.session_state['subtitles_added'] = True

    if st.session_state.get('subtitles_added', False):
        with open(f"video_with_subtitles{now}.mp4", "rb") as file:
            st.download_button(
                label="下载带字幕的视频",
                data=file,
                file_name=f"video_with_subtitles{now}.mp4",
                mime="video/mp4"
            )


if __name__ == "__main__":
    main()
