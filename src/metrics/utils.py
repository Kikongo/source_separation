# Based on seminar materials

# Don't forget to support cases when target_text == ''
import editdistance

def calc_cer(target_text, predicted_text) -> float:
    #assert len(target_text) > 0 or len(predicted_text) > 0, "Both target and predicted texts are empty."

    return editdistance.eval(target_text, predicted_text) / len(target_text) if target_text != '' else 1.0


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    #assert len(target_words) > 0 or len(predicted_words) > 0, "Both target and predicted texts are empty."

    return editdistance.eval(target_words, predicted_words) / len(target_words) if len(target_words) > 0 else 1.0