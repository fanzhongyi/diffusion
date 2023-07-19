import json
import logging


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling dataset error ({repr(exn)}). Ignoring.')
    return True


def long_short_ratio(a, b):
    return max(a, b) / min(a, b)


def filter_fn(sample, filter_strategy=None):
    ''' filter the sample that not satisfies filter_strategy '''
    ''' the sample should be a dict type that contains the json key'''

    # if no filter requirements, then return True to keep this sample
    if filter_strategy is None:
        return True

    dic = sample
    if isinstance(dic, bytes):
        dic = json.loads(dic)

    if 'filter_by_short_side' in filter_strategy:
        threshold = filter_strategy["filter_by_short_side"].get("threshold", 10)
        width = dic.get("width") or dic.get("img_params",
                                            {}).get("width") or dic.get("WIDTH")
        height = dic.get("height") or dic.get(
            "img_params", {}).get("height") or dic.get("HEIGHT")

        if width is not None and height is not None and min(width,
                                                            height) < threshold:
            return False

        if width is None or height is None and not filter_strategy[
                'filter_by_short_side']["default"]:
            return False

    if "long_short_ratio" in filter_strategy:
        threshold = filter_strategy["long_short_ratio"].get(
            "threshold", float("inf"))
        width = dic.get("WIDTH") or dic.get("img_params",
                                            {}).get("width") or dic.get("width")
        height = dic.get("HEIGHT") or dic.get(
            "img_params", {}).get("height") or dic.get("height")

        if width is not None and height is not None and long_short_ratio(
                width, height) > threshold:
            return False

        if width is None or height is None and not filter_strategy[
                "long_short_ratio"]["default"]:
            return False

    if "punsafe" in filter_strategy:
        threshold = filter_strategy["punsafe"].get("threshold", float("inf"))
        punsafe_score = dic.get("punsafe") or dic.get(
            "meta", {}).get("punsafe") or dic.get(
                "add_meta",
                {}).get("punsafe") or dic.get("nsfw_score_opennsfw2")

        if punsafe_score is not None and punsafe_score > threshold:
            return False

        if punsafe_score is None and not filter_strategy['punsafe']["default"]:
            return False

    if "aesthetic" in filter_strategy:
        threshold = filter_strategy["aesthetic"].get("threshold", float("-inf"))
        aesthetic_score = dic.get("aesthetic_score_laion_v2") or dic.get(
            "AESTHETIC_SCORE") or dic.get(
                "meta", {}).get("aesthetic_score") or dic.get(
                    "add_meta", {}).get("aesthetic_score")

        if aesthetic_score is not None and aesthetic_score < threshold:
            return False

        if aesthetic_score is None and not filter_strategy['aesthetic'][
                "default"]:
            return False

    if "pwatermark" in filter_strategy:
        threshold = filter_strategy["pwatermark"].get("threshold", float("inf"))
        pwatermark_score = dic.get("pwatermark") or dic.get(
            "meta", {}).get("pwatermark") or dic.get(
                "add_meta", {}).get("pwatermark") or dic.get("watermark_score")

        if pwatermark_score is not None and pwatermark_score > threshold:
            return False

        if pwatermark_score is None and not filter_strategy['pwatermark'][
                "default"]:
            return False

    return True
