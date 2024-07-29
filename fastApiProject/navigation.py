#navigation.py
import speech_recognition as sr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recognize_speech_from_mic(recognizer, microphone_index=None):
    if microphone_index is not None:
        microphone = sr.Microphone(device_index=microphone_index)
    else:
        microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        logger.info("Listening...")
        audio = recognizer.listen(source)

    logger.info("Recognizing...")
    try:
        response = recognizer.recognize_google(audio, language='ko-KR')
        logger.info(f"You said: {response}")
        return response
    except sr.UnknownValueError:
        logger.error("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def get_destination_from_voice(max_attempts=3, microphone_index=None):
    recognizer = sr.Recognizer()

    for attempt in range(max_attempts):
        logger.info("Please say the destination location (Attempt {}/{}):".format(attempt + 1, max_attempts))
        destination = recognize_speech_from_mic(recognizer, microphone_index)
        if destination:
            return destination
        logger.error("Failed to recognize the destination location. Trying again...")

    logger.error("Exceeded maximum attempts to recognize destination location.")
    return None
