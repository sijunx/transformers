from huggingsound import SpeechRecognitionModel
model = SpeechRecognitionModel("wbbbbb/wav2vec2-large-chinese-zh-cn")
# audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]
# audio_paths = ["/Users/xusijun/Documents/常用文件/录音转文本/test.mp3"]
audio_paths = ["/Users/zard/Documents/nlp_data/changer.mp3"]
transcriptions = model.transcribe(audio_paths)

print(transcriptions)
