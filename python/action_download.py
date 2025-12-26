from configparser import ConfigParser
from typing import Any

from src.screening.xsa import HttpCurlCrawler

from common import OBS_ID_COLS, FILTER_COLS
from common import extract_row_value 


def action_download(
    config: ConfigParser,
    input_row: dict[str, Any],
) -> None:
    section = config['INPUT']

    crawler: HttpCurlCrawler = HttpCurlCrawler(
        download_directory=section['DOWNLOAD_DIRECTORY'],
        base_url=section['BASE_URL'],
        regex_pattern=section['REGEX'],
    )

    crawler.crawl(
        observation_id=extract_row_value(input_row, OBS_ID_COLS),
        filters=[extract_row_value(input_row, FILTER_COLS)],
    )
