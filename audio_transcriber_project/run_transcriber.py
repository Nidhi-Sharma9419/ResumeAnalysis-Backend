from AudioTranscriber import AudioTranscriber

def main():
    transcriber = AudioTranscriber()
    
    # Replace with your audio file (WAV format recommended)
    audio_file = "audio2.wav"  
    
    # Run analysis
    results = transcriber.analyze_audio(audio_file)
    
    # Print results
    print("\n=== Transcription ===")
    print(results["transcript"])
    
    print("\n=== Speech Analysis ===")
    print(f"Speech Rate: {results['speech_rate']} words/min")
    print(f"Audio Duration: {results['audio_duration']} sec")
    print(f"Word Count: {results['word_count']}")
    
    print("\n=== Filler Words ===")
    print(f"Total Fillers: {results['filler_analysis']['total_fillers']}")
    print(f"Filler Words Used: {results['filler_analysis']['filler_words']}")
    print(f"Filler Frequency: {results['filler_analysis']['filler_frequency']:.2f} fillers/word")
    
    print("\n=== Silent Gaps ===")
    print(f"Total Gaps: {results['gap_count']}")
    for gap in results['gap_analysis']:
        print(f"  - Gap from {gap['start']}s to {gap['end']}s (Duration: {gap['duration']}s)")

if __name__ == "__main__":
    main()