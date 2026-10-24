# OCR eval photos (fetched, not committed)

`server/tools/ocr_eval.py` (and `tests/test_text_read_quality.py`) download these Wikimedia Commons originals here on
first use and check each against the sha1 Commons publishes. The JPEGs are gitignored: they are ~35 MB, and this
repository does not redistribute them. They are used unmodified, for local evaluation only.

| file | Commons page | author | licence |
|---|---|---|---|
| `Emergency_exit_sign_01.jpg` | [File:Emergency exit sign 01.jpg](https://commons.wikimedia.org/wiki/File:Emergency_exit_sign_01.jpg) | Kritzolina | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `Taiwan_exit_sign.jpg` | [File:Taiwan exit sign white backgrounded.jpg](https://commons.wikimedia.org/wiki/File:Taiwan_exit_sign_white_backgrounded.jpg) | Oakw | Public domain |
| `Tokyo_Station_sign.jpg` | [File:Tokyo Station Sign (Yamanote Line).jpg](https://commons.wikimedia.org/wiki/File:Tokyo_Station_Sign_(Yamanote_Line).jpg) | そらみみ | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) |
| `Moscow_Arbatskaya_sign.jpg` | [File:Moscow, Arbatskaya (APL) metro station - sign (43315108441).jpg](https://commons.wikimedia.org/wiki/File:Moscow,_Arbatskaya_(APL)_metro_station_-_sign_(43315108441).jpg) | Andrey Filippov | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) |
| `Sidi_El_Wasti_sign.jpg` | [File:Arabic-English street sign "Sidi El Wasti St." in Alexandria.jpg](https://commons.wikimedia.org/wiki/File:Arabic-English_street_sign_%22Sidi_El_Wasti_St.%22_in_Alexandria.jpg) | Marsupium | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `Del_Taco_menu.jpg` | [File:Drive thru menu board at Del Taco in Albany, Oregon.jpg](https://commons.wikimedia.org/wiki/File:Drive_thru_menu_board_at_Del_Taco_in_Albany,_Oregon.jpg) | Rick Obst | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `HK_paper_menus.jpg` | [File:Paper menus and white board menu in the hk style restaurants.jpg](https://commons.wikimedia.org/wiki/File:Paper_menus_and_white_board_menu_in_the_hk_style_restaurants.jpg) | Peachyeung316 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `Piwna_Street.jpg` | [File:Piwna Street, Warsaw 05.jpg](https://commons.wikimedia.org/wiki/File:Piwna_Street,_Warsaw_05.jpg) | Simon Burchell | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `Laptop_code.jpg` | [File:Macro laptop coding (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Macro_laptop_coding_(Unsplash).jpg) | Marc Mueller | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |

`bus.jpg` comes with the ultralytics package (`ultralytics/assets`).
