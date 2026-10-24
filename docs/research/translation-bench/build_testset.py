"""Builds testset.jsonl: realistic OCR'd sign / menu / notice / document texts with reference translations.

Every item: id, src, tgt, cat, kind, text (lines joined with "\n"), refs (1+ acceptable references, first = primary),
accept (lower-case key terms; a hit = any of them appears in the output; only for signs where one term decides
adequacy), ctx (image-caption-like hint, optional), nb (neighbouring blocks on the same photo, optional).

cat: word (one-word sign), short (2-5 word sign), caps (ALL CAPS sign), multi (multi-line sign), warn (warning /
notice), menu, time (timetable / opening hours), doc (document / notice prose), ui (screen), label (product /
medicine label), noise (OCR-noise variant of a sign), codes (numbers/codes mixed in), ff (false friend / needs context).
kind: prose = translate the block as one text; list = one entry per line (menus, timetables, UI lists).

References were written by hand (see README.md for how they were made and checked).
Run: python build_testset.py  (writes testset.jsonl next to this file)
"""

import json
from pathlib import Path

ITEMS: list[dict] = []


def E(src, tgt, cat, text, refs, kind=None, acc=None, ctx=None, nb=None):
    refs = [refs] if isinstance(refs, str) else list(refs)
    if kind is None:
        kind = "list" if cat in ("menu", "time", "ui") else "prose"
    n = sum(1 for i in ITEMS if i["src"] == src and i["tgt"] == tgt) + 1
    item = {"id": f"{src}-{tgt}-{n:02d}", "src": src, "tgt": tgt, "cat": cat, "kind": kind, "text": text, "refs": refs}
    if acc:
        item["accept"] = [a.lower() for a in acc]
    if ctx:
        item["ctx"] = ctx
    if nb:
        item["nb"] = nb
    ITEMS.append(item)


EXIT_EN = ["exit", "way out"]
ENTRANCE_EN = ["entrance", "entry", "way in"]
EMERG_EN = ["emergency exit", "fire exit"]
NOSMOKE_EN = ["no smoking", "smoking prohibited", "smoking is prohibited", "smoking forbidden", "smoking is forbidden",
              "smoking not allowed", "do not smoke"]
NOENTRY_EN = ["no entry", "keep out", "no trespassing", "entry prohibited", "entry forbidden", "do not enter",
              "no admittance", "entry is prohibited", "entry is forbidden", "access prohibited", "no access"]
TOILET_EN = ["toilet", "restroom", "wc", "lavatory", "bathroom", "washroom"]
PUSH_EN, PULL_EN = ["push"], ["pull"]
WYJ = ["wyjście"]
NOSMOKE_PL = ["zakaz palenia", "palenie wzbronione", "nie palić", "palenie zabronione"]
NOENTRY_PL = ["wstęp wzbroniony", "zakaz wstępu", "wejście wzbronione", "wstęp zabroniony", "nie wchodzić"]
CLOSED_PL = ["zamknięte", "nieczynne", "nieczynny", "zamknięty"]
EMERG_PL = ["wyjście ewakuacyjne", "wyjście awaryjne", "wyjście bezpieczeństwa"]
DOOR = "sticker on a glass door"
OVERDOOR = "sign above a door"
STATION = "sign in a railway station"

# ---------------------------------------------------------------- German
E("de", "en", "word", "Ausgang", ["Exit", "Way out"], acc=EXIT_EN, ctx=STATION, nb=["Gleis 1–4", "U-Bahn"])
E("de", "en", "word", "Eingang", ["Entrance", "Way in"], acc=ENTRANCE_EN, ctx=OVERDOOR)
E("de", "en", "word", "Notausgang", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("de", "en", "caps", "NOTAUSGANG", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("de", "en", "word", "Ziehen", ["Pull"], acc=PULL_EN, ctx=DOOR)
E("de", "en", "word", "Drücken", ["Push"], acc=PUSH_EN, ctx=DOOR)
E("de", "en", "short", "Rauchen verboten", ["No smoking", "Smoking prohibited"], acc=NOSMOKE_EN)
E("de", "en", "caps", "BETRETEN VERBOTEN", ["No entry", "Keep out", "No trespassing"], acc=NOENTRY_EN)
E("de", "en", "multi", "Achtung!\nRutschgefahr bei Nässe", ["Caution!\nSlippery when wet"], acc=["slipp"])
E("de", "en", "warn", "Bitte nicht berühren", ["Please do not touch"], acc=["not touch", "don't touch"])
E("de", "en", "menu", "Speisekarte\nSchnitzel Wiener Art mit Pommes 14,90 €\nBratwurst mit Sauerkraut 9,50 €\n"
  "Apfelstrudel mit Sahne 5,80 €",
  ["Menu\nViennese-style schnitzel with fries 14,90 €\nBratwurst with sauerkraut 9,50 €\n"
   "Apple strudel with cream 5,80 €"], ctx="restaurant menu")
E("de", "en", "time", "Öffnungszeiten\nMo–Fr 9–18 Uhr\nSa 10–14 Uhr\nSonn- und Feiertage geschlossen",
  ["Opening hours\nMon–Fri 9 am–6 pm\nSat 10 am–2 pm\nClosed on Sundays and public holidays",
   "Opening hours\nMon–Fri 9:00–18:00\nSat 10:00–14:00\nClosed Sundays and public holidays"], ctx="shop door")
E("de", "en", "doc", "Der Fahrausweis ist vor Fahrtantritt zu entwerten. Bei Kontrollen ohne gültigen Fahrschein "
  "wird ein erhöhtes Beförderungsentgelt von 60 € erhoben.",
  ["The ticket must be validated before starting the journey. Passengers found without a valid ticket during "
   "inspections will be charged a penalty fare of €60."], ctx="notice in a tram")
E("de", "en", "noise", "AUSGANC", ["Exit"], acc=EXIT_EN, ctx=OVERDOOR)
E("de", "en", "short", "Gleis 3", ["Platform 3", "Track 3"], acc=["platform 3", "track 3"], ctx=STATION)
E("de", "en", "label", "Kühl und trocken lagern.\nMindestens haltbar bis: siehe Deckel",
  ["Store in a cool, dry place.\nBest before: see lid"], ctx="food packaging")
E("de", "en", "ui", "Einstellungen\nAbbrechen\nSpeichern", ["Settings\nCancel\nSave"], ctx="phone screen")
E("de", "en", "short", "Kein Trinkwasser", ["Not drinking water", "Non-potable water"],
  acc=["not drinking water", "non-potable", "no drinking water", "not potable", "not for drinking", "undrinkable"],
  ctx="label above a tap")
E("de", "en", "codes", "Ausfahrt 23 Köln-Ost\n2 km", ["Exit 23 Köln-Ost\n2 km", "Exit 23 Cologne-East\n2 km"],
  acc=["exit 23"], ctx="motorway sign")
E("de", "en", "ff", "Gift", ["Poison", "Toxic"], acc=["poison", "toxic"], ctx="label on a bottle")
E("de", "en", "ff", "Frei", ["Vacant", "Free"], acc=["vacant", "free", "unoccupied", "available"],
  ctx="indicator on a toilet door lock")
E("de", "en", "multi", "Ausgang / Exit", ["Exit"], acc=EXIT_EN, ctx=OVERDOOR)
E("de", "pl", "word", "Ausgang", "Wyjście", acc=WYJ, ctx=STATION)
E("de", "pl", "word", "Notausgang", ["Wyjście ewakuacyjne", "Wyjście awaryjne"], acc=EMERG_PL, ctx=OVERDOOR)
E("de", "pl", "short", "Rauchen verboten", ["Zakaz palenia", "Palenie wzbronione"], acc=NOSMOKE_PL)
E("de", "pl", "short", "Bitte Abstand halten", ["Prosimy zachować odstęp", "Proszę zachować odstęp"],
  acc=["odstęp", "dystans", "odległość"])
E("de", "pl", "short", "Vorsicht Stufe", ["Uwaga, stopień", "Uwaga na stopień"], acc=["stop"])
E("de", "pl", "word", "Apotheke", "Apteka", acc=["apteka"])
E("de", "pl", "word", "Kasse", ["Kasa"], acc=["kasa"], ctx="sign in a supermarket")
E("de", "pl", "doc", "Das Parken ist nur mit gültigem Parkschein erlaubt. Widerrechtlich abgestellte Fahrzeuge "
  "werden kostenpflichtig abgeschleppt.",
  ["Parkowanie dozwolone wyłącznie z ważnym biletem parkingowym. Pojazdy zaparkowane niezgodnie z przepisami "
   "zostaną odholowane na koszt właściciela."], ctx="sign in a car park")
E("de", "pl", "menu", "Frühstück\nRührei mit Speck\nMüsli mit Joghurt", ["Śniadanie\nJajecznica z boczkiem\n"
  "Musli z jogurtem"], ctx="hotel breakfast menu")
E("de", "pl", "caps", "BETRETEN DER BAUSTELLE VERBOTEN", ["Wstęp na teren budowy wzbroniony",
  "Zakaz wstępu na teren budowy"], acc=["wzbron", "zakaz", "zabron"], ctx="fence of a construction site")
E("de", "es", "word", "Ausgang", "Salida", acc=["salida"], ctx=STATION)

# ---------------------------------------------------------------- Polish
E("pl", "en", "word", "Wyjście", ["Exit", "Way out"], acc=EXIT_EN, ctx=OVERDOOR)
E("pl", "en", "word", "Wejście", ["Entrance", "Way in"], acc=ENTRANCE_EN, ctx=OVERDOOR)
E("pl", "en", "caps", "WYJŚCIE EWAKUACYJNE", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("pl", "en", "short", "Zakaz palenia", ["No smoking"], acc=NOSMOKE_EN)
E("pl", "en", "multi", "Uwaga!\nŚwieżo malowane", ["Caution!\nWet paint", "Attention!\nFreshly painted"],
  acc=["wet paint", "fresh paint", "freshly painted"])
E("pl", "en", "word", "Ciągnąć", ["Pull"], acc=PULL_EN, ctx=DOOR)
E("pl", "en", "word", "Pchać", ["Push"], acc=PUSH_EN, ctx=DOOR)
E("pl", "en", "short", "Kasa nieczynna", ["Checkout closed", "Till closed", "Cash desk closed"], acc=["closed"],
  ctx="sign at a supermarket checkout")
E("pl", "en", "menu", "Zupy\nŻurek w chlebie 18 zł\nRosół z makaronem 14 zł\nPierogi ruskie 24 zł",
  ["Soups\nSour rye soup in a bread bowl 18 zł\nChicken broth with noodles 14 zł\n"
   "Pierogi with potato and cheese 24 zł",
   "Soups\nŻurek in bread 18 zł\nBroth with noodles 14 zł\nRuthenian dumplings 24 zł"], ctx="restaurant menu")
E("pl", "en", "time", "Godziny otwarcia\npon.–pt. 8:00–20:00\nsob. 9:00–15:00\nniedz. nieczynne",
  ["Opening hours\nMon–Fri 8:00–20:00\nSat 9:00–15:00\nSun closed"], ctx="shop door")
E("pl", "en", "doc", "Bilet należy skasować niezwłocznie po wejściu do pojazdu. Brak ważnego biletu podlega "
  "opłacie dodatkowej.",
  ["The ticket must be validated immediately after boarding the vehicle. Travelling without a valid ticket is "
   "subject to a surcharge."], ctx="notice in a bus")
E("pl", "en", "noise", "Wyjśc1e", ["Exit"], acc=EXIT_EN, ctx=OVERDOOR)
E("pl", "en", "noise", "ZAKAZ PALENlA", ["No smoking"], acc=NOSMOKE_EN)
E("pl", "en", "ui", "Ustawienia\nAnuluj\nZapisz zmiany", ["Settings\nCancel\nSave changes"], ctx="phone screen")
E("pl", "en", "label", "Przechowywać w miejscu niedostępnym dla dzieci.",
  ["Keep out of the reach of children.", "Store out of the reach of children."], acc=["children"],
  ctx="medicine box")
E("pl", "en", "short", "Teren prywatny", ["Private property", "Private area"], acc=["private"])
E("pl", "en", "word", "Toaleta", ["Toilet", "Restroom", "WC"], acc=TOILET_EN)
E("pl", "en", "ff", "Pokój 12", ["Room 12"], acc=["room 12"], ctx="hotel corridor door")
E("pl", "en", "warn", "Uwaga! Zły pies", ["Beware of the dog!", "Caution! Vicious dog"], acc=["dog"])
E("pl", "en", "doc", "Szanowni Klienci, w dniu 11 listopada sklep będzie nieczynny. Zapraszamy od 12 listopada.",
  ["Dear Customers, the shop will be closed on 11 November. We look forward to seeing you from 12 November."],
  ctx="notice on a shop door")
E("pl", "de", "word", "Wyjście", "Ausgang", acc=["ausgang"], ctx=OVERDOOR)
E("pl", "de", "short", "Zakaz wstępu", ["Zutritt verboten", "Kein Zutritt", "Betreten verboten"],
  acc=["verboten", "kein zutritt", "untersagt"])
E("pl", "de", "warn", "Uwaga! Zły pies", ["Vorsicht, bissiger Hund!", "Achtung! Bissiger Hund"], acc=["hund"])
E("pl", "es", "caps", "WYJŚCIE EWAKUACYJNE", ["Salida de emergencia"], acc=["salida de emergencia", "salida de evacuación"])
E("pl", "es", "time", "Godziny otwarcia\npon.–pt. 10–18", ["Horario de apertura\nlun.–vie. 10–18",
  "Horario\nlunes a viernes 10:00–18:00"], ctx="shop door")

# ---------------------------------------------------------------- French
E("fr", "en", "word", "Sortie", ["Exit", "Way out"], acc=EXIT_EN, ctx=STATION, nb=["Métro", "Quais 1 à 6"])
E("fr", "en", "caps", "SORTIE DE SECOURS", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("fr", "en", "word", "Poussez", ["Push"], acc=PUSH_EN, ctx=DOOR)
E("fr", "en", "word", "Tirez", ["Pull"], acc=PULL_EN, ctx=DOOR)
E("fr", "en", "short", "Défense de fumer", ["No smoking"], acc=NOSMOKE_EN)
E("fr", "en", "short", "Attention à la marche", ["Mind the step", "Watch your step"], acc=["step"])
E("fr", "en", "short", "Fermé le lundi", ["Closed on Mondays", "Closed Mondays"], acc=["closed"])
E("fr", "en", "menu", "Entrées\nSoupe à l'oignon gratinée 8 €\nEscargots de Bourgogne (6 pièces) 12 €\nPlats\n"
  "Confit de canard, pommes sarladaises 22 €",
  ["Starters\nFrench onion soup au gratin 8 €\nBurgundy snails (6 pieces) 12 €\nMain courses\n"
   "Duck confit, Sarlat-style potatoes 22 €"], ctx="restaurant menu")
E("fr", "en", "doc", "Il est interdit de jeter des objets par la fenêtre. Tout abus sera sanctionné.",
  ["It is forbidden to throw objects out of the window. Any misuse will be penalised."], ctx="notice in a train")
E("fr", "en", "noise", "S0RTIE", ["Exit"], acc=EXIT_EN, ctx=OVERDOOR)
E("fr", "en", "ui", "Paramètres\nAnnuler\nEnregistrer", ["Settings\nCancel\nSave"], ctx="phone screen")
E("fr", "en", "warn", "Chien méchant", ["Beware of the dog", "Vicious dog"], acc=["dog"], ctx="sign on a garden gate")
E("fr", "en", "word", "Toilettes", ["Toilets", "Restrooms"], acc=TOILET_EN)
E("fr", "en", "ff", "Libre", ["Vacant", "Free"], acc=["vacant", "free", "unoccupied", "available"],
  ctx="indicator on a toilet door lock")
E("fr", "en", "time", "Horaires d'ouverture\nDu lundi au vendredi : 9h – 19h\nSamedi : 9h – 12h",
  ["Opening hours\nMonday to Friday: 9 am – 7 pm\nSaturday: 9 am – 12 pm",
   "Opening hours\nMonday to Friday: 9:00 – 19:00\nSaturday: 9:00 – 12:00"], ctx="pharmacy door")
E("fr", "pl", "word", "Sortie", "Wyjście", acc=WYJ, ctx=OVERDOOR)
E("fr", "pl", "short", "Défense de fumer", ["Zakaz palenia"], acc=NOSMOKE_PL)
E("fr", "pl", "short", "Fermé le dimanche", ["Nieczynne w niedziele", "W niedziele zamknięte"],
  acc=["niedziel"])
E("fr", "pl", "short", "Interdit au public", ["Wstęp tylko dla upoważnionych", "Zakaz wstępu dla osób postronnych"],
  acc=["wstęp", "wstępu", "postronn", "nieupoważnion", "upoważnion"])
E("fr", "pl", "warn", "Attention, chute de pierres", ["Uwaga, spadające kamienie"], acc=["kamie", "skał"])
E("fr", "de", "caps", "SORTIE DE SECOURS", ["Notausgang"], acc=["notausgang"], ctx=OVERDOOR)

# ---------------------------------------------------------------- Spanish
E("es", "en", "word", "Salida", ["Exit", "Way out"], acc=EXIT_EN, ctx=OVERDOOR)
E("es", "en", "word", "Empujar", ["Push"], acc=PUSH_EN, ctx=DOOR)
E("es", "en", "word", "Tirar", ["Pull"], acc=PULL_EN, ctx=DOOR)
E("es", "en", "short", "Prohibido fumar", ["No smoking"], acc=NOSMOKE_EN)
E("es", "en", "short", "Salida de emergencia", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("es", "en", "caps", "CUIDADO: SUELO MOJADO", ["Caution: wet floor"], acc=["wet floor", "slippery"])
E("es", "en", "short", "Se alquila", ["For rent", "To let"], acc=["for rent", "to let", "for lease", "rent"],
  ctx="sign in a shop window")
E("es", "en", "word", "Aseos", ["Toilets", "Restrooms"], acc=TOILET_EN)
E("es", "en", "menu", "Tapas\nPatatas bravas 4,50 €\nCroquetas de jamón 6,00 €\nPulpo a la gallega 14,00 €",
  ["Tapas\nSpicy fried potatoes 4,50 €\nHam croquettes 6,00 €\nGalician-style octopus 14,00 €",
   "Tapas\nPatatas bravas 4,50 €\nHam croquettes 6,00 €\nOctopus Galician style 14,00 €"], ctx="bar menu")
E("es", "en", "doc", "Prohibido el paso a toda persona ajena a la obra.",
  ["No entry for unauthorised persons on the construction site.",
   "Access prohibited to anyone not involved in the works."], ctx="construction site fence")
E("es", "en", "noise", "Sal1da", ["Exit"], acc=EXIT_EN, ctx=OVERDOOR)
E("es", "en", "ff", "Caja", ["Checkout", "Cashier", "Till"], acc=["checkout", "cashier", "till", "cash desk", "register"],
  ctx="sign in a supermarket")
E("es", "en", "time", "Horario\nLunes a viernes: 10:00 - 14:00 y 17:00 - 20:30\nSábados: 10:00 - 14:00",
  ["Opening hours\nMonday to Friday: 10:00 - 14:00 and 17:00 - 20:30\nSaturdays: 10:00 - 14:00"],
  ctx="shop door")
E("es", "pl", "word", "Salida", "Wyjście", acc=WYJ, ctx=OVERDOOR)
E("es", "pl", "short", "Prohibido fumar", ["Zakaz palenia"], acc=NOSMOKE_PL)
E("es", "pl", "short", "Cerrado por vacaciones", ["Zamknięte z powodu urlopu", "Nieczynne – urlop"],
  acc=["urlop", "wakacj", "ferie"])
E("es", "pl", "caps", "PELIGRO: ALTA TENSIÓN", ["Niebezpieczeństwo: wysokie napięcie"], acc=["napięci"])

# ---------------------------------------------------------------- Italian
E("it", "en", "word", "Uscita", ["Exit", "Way out"], acc=EXIT_EN, ctx=OVERDOOR)
E("it", "en", "word", "Ingresso", ["Entrance", "Way in"], acc=ENTRANCE_EN, ctx=OVERDOOR)
E("it", "en", "short", "Vietato fumare", ["No smoking"], acc=NOSMOKE_EN)
E("it", "en", "caps", "USCITA DI SICUREZZA", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("it", "en", "word", "Spingere", ["Push"], acc=PUSH_EN, ctx=DOOR)
E("it", "en", "short", "Chiuso per ferie", ["Closed for holidays", "Closed for vacation"], acc=["closed"])
E("it", "en", "menu", "Primi piatti\nSpaghetti alla carbonara 12 €\nRisotto ai funghi porcini 15 €\nSecondi\n"
  "Cotoletta alla milanese 18 €",
  ["First courses\nSpaghetti carbonara 12 €\nPorcini mushroom risotto 15 €\nMain courses\n"
   "Milanese-style breaded veal cutlet 18 €"], ctx="restaurant menu")
E("it", "en", "doc", "Si prega di convalidare il biglietto prima di salire a bordo.",
  ["Please validate your ticket before boarding."], ctx="notice at a station")
E("it", "en", "noise", "VIETAT0 FUMARE", ["No smoking"], acc=NOSMOKE_EN)
E("it", "en", "ff", "Caldo", ["Hot"], acc=["hot", "warm"], ctx="label on a tap")
E("it", "pl", "word", "Uscita", "Wyjście", acc=WYJ, ctx=OVERDOOR)
E("it", "pl", "short", "Vietato l'ingresso", ["Zakaz wstępu", "Wstęp wzbroniony"], acc=NOENTRY_PL)
E("it", "pl", "short", "Lavori in corso", ["Roboty drogowe", "Trwają prace"], acc=["robot", "prac", "remont"])

# ---------------------------------------------------------------- Russian
E("ru", "en", "word", "Выход", ["Exit", "Way out"], acc=EXIT_EN, ctx=OVERDOOR)
E("ru", "en", "word", "Вход", ["Entrance", "Way in"], acc=ENTRANCE_EN, ctx=OVERDOOR)
E("ru", "en", "caps", "ВХОД ВОСПРЕЩЁН", ["No entry", "Entry prohibited"], acc=NOENTRY_EN)
E("ru", "en", "short", "Не курить", ["No smoking", "Do not smoke"], acc=NOSMOKE_EN)
E("ru", "en", "warn", "Осторожно, двери закрываются", ["Caution, the doors are closing", "Mind the closing doors"],
  acc=["door"], ctx="sticker on a metro train door")
E("ru", "en", "short", "Запасный выход", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("ru", "en", "ff", "Касса", ["Ticket office", "Box office"], acc=["ticket office", "box office", "tickets",
  "cash desk", "ticket"], ctx=STATION, nb=["Пригородные поезда"])
E("ru", "en", "menu", "Меню\nБорщ со сметаной 350 ₽\nПельмени домашние 420 ₽\nБлины с икрой 690 ₽",
  ["Menu\nBorscht with sour cream 350 ₽\nHomemade pelmeni 420 ₽\nBlini with caviar 690 ₽",
   "Menu\nBorsch with sour cream 350 ₽\nHomemade dumplings 420 ₽\nPancakes with caviar 690 ₽"],
  ctx="restaurant menu")
E("ru", "en", "time", "Режим работы\nПн–Пт 9:00–21:00\nСб–Вс 10:00–18:00\nБез перерыва",
  ["Opening hours\nMon–Fri 9:00–21:00\nSat–Sun 10:00–18:00\nNo lunch break",
   "Opening hours\nMon–Fri 9:00–21:00\nSat–Sun 10:00–18:00\nOpen without a break"], ctx="shop door")
E("ru", "en", "doc", "Уважаемые пассажиры! При обнаружении подозрительных предметов сообщите машинисту поезда.",
  ["Dear passengers! If you notice any suspicious objects, please inform the train driver."],
  ctx="notice in a metro car")
E("ru", "en", "noise", "BЫXOД", ["Exit"], acc=EXIT_EN, ctx=OVERDOOR)
E("ru", "en", "short", "Выход в город", ["Exit to the city", "Way out to the street"], acc=["exit", "way out"],
  ctx="sign in a metro station")
E("ru", "en", "label", "Хранить при температуре не выше 25 °C.\nБеречь от детей.",
  ["Store at a temperature not exceeding 25 °C.\nKeep away from children.",
   "Store below 25 °C.\nKeep out of the reach of children."], ctx="medicine box")
E("ru", "pl", "word", "Выход", "Wyjście", acc=WYJ, ctx=OVERDOOR)
E("ru", "pl", "short", "Не курить", ["Nie palić", "Zakaz palenia"], acc=NOSMOKE_PL)
E("ru", "pl", "caps", "ВХОД ВОСПРЕЩЁН", ["Wstęp wzbroniony", "Zakaz wstępu"], acc=NOENTRY_PL)
E("ru", "pl", "warn", "Осторожно, злая собака", ["Uwaga, zły pies"], acc=["pies", "psa", "psem"])
E("ru", "pl", "word", "Аптека", "Apteka", acc=["apteka"])
E("ru", "pl", "codes", "Перерыв с 13:00 до 14:00", ["Przerwa od 13:00 do 14:00", "Przerwa 13:00–14:00"],
  acc=["przerw"], ctx="shop door")
E("ru", "de", "word", "Выход", "Ausgang", acc=["ausgang"], ctx=OVERDOOR)

# ---------------------------------------------------------------- Ukrainian
E("uk", "en", "word", "Вихід", ["Exit", "Way out"], acc=EXIT_EN, ctx=OVERDOOR)
E("uk", "en", "word", "Вхід", ["Entrance", "Way in"], acc=ENTRANCE_EN, ctx=OVERDOOR)
E("uk", "en", "caps", "ПАЛИТИ ЗАБОРОНЕНО", ["No smoking", "Smoking is prohibited"], acc=NOSMOKE_EN)
E("uk", "en", "multi", "Обережно!\nСлизька підлога", ["Caution!\nSlippery floor"], acc=["slipp", "wet floor"])
E("uk", "en", "word", "Зачинено", ["Closed"], acc=["closed"], ctx="sign on a shop door")
E("uk", "en", "ff", "Каса", ["Ticket office", "Box office"], acc=["ticket office", "box office", "tickets",
  "cash desk", "ticket"], ctx=STATION)
E("uk", "en", "short", "Аварійний вихід", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("uk", "en", "doc", "Шановні пасажири! Не залишайте свої речі без нагляду.",
  ["Dear passengers! Do not leave your belongings unattended."], ctx="notice at a station")
E("uk", "en", "menu", "Меню\nВареники з картоплею 120 грн\nБорщ український 95 грн\nДеруни зі сметаною 110 грн",
  ["Menu\nVarenyky with potatoes 120 UAH\nUkrainian borscht 95 UAH\nPotato pancakes with sour cream 110 UAH",
   "Menu\nPotato dumplings 120 грн\nUkrainian borsch 95 грн\nDeruny with sour cream 110 грн"],
  ctx="restaurant menu")
E("uk", "pl", "word", "Вихід", "Wyjście", acc=WYJ, ctx=OVERDOOR)
E("uk", "pl", "word", "Зачинено", ["Zamknięte", "Nieczynne"], acc=CLOSED_PL, ctx="sign on a shop door")
E("uk", "pl", "short", "Стороннім вхід заборонено", ["Osobom postronnym wstęp wzbroniony",
  "Zakaz wstępu dla osób nieupoważnionych"], acc=["wzbron", "zakaz", "zabron"])
E("uk", "pl", "short", "Обережно, сходинка", ["Uwaga, stopień", "Uwaga na stopień"], acc=["stop"])
E("uk", "pl", "word", "Відчинено", ["Otwarte", "Czynne"], acc=["otwart", "czynne"], ctx="sign on a shop door")

# ---------------------------------------------------------------- Japanese
E("ja", "en", "word", "出口", ["Exit", "Way out"], acc=EXIT_EN, ctx=STATION, nb=["中央改札", "1番線"])
E("ja", "en", "word", "入口", ["Entrance", "Way in"], acc=ENTRANCE_EN, ctx=OVERDOOR)
E("ja", "en", "word", "非常口", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("ja", "en", "word", "禁煙", ["No smoking", "Non-smoking"], acc=NOSMOKE_EN + ["non-smoking"])
E("ja", "en", "word", "押す", ["Push"], acc=PUSH_EN, ctx=DOOR)
E("ja", "en", "word", "引く", ["Pull"], acc=PULL_EN, ctx=DOOR)
E("ja", "en", "word", "立入禁止", ["No entry", "Keep out"], acc=NOENTRY_EN)
E("ja", "en", "word", "営業中", ["Open", "Now open"], acc=["open"], ctx="sign hanging on a restaurant door")
E("ja", "en", "warn", "足元にご注意ください", ["Please watch your step", "Please mind your step"], acc=["step", "footing"])
E("ja", "en", "word", "お手洗い", ["Restroom", "Toilet"], acc=TOILET_EN)
E("ja", "en", "menu", "メニュー\nラーメン 850円\n餃子（6個） 400円\n生ビール 550円",
  ["Menu\nRamen 850 yen\nGyoza (6 pieces) 400 yen\nDraft beer 550 yen",
   "Menu\nRamen ¥850\nDumplings (6 pcs) ¥400\nDraught beer ¥550"], ctx="restaurant menu")
E("ja", "en", "time", "営業時間\n11:00〜22:00\n定休日 水曜日",
  ["Business hours\n11:00–22:00\nClosed on Wednesdays", "Opening hours\n11:00–22:00\nRegular holiday: Wednesday"],
  ctx="restaurant door")
E("ja", "en", "doc", "駆け込み乗車は危険ですのでおやめください。",
  ["Rushing onto the train is dangerous, so please refrain from doing so.",
   "Please do not rush onto the train; it is dangerous."], ctx="poster on a station platform")
E("ja", "en", "noise", "出囗", ["Exit"], acc=EXIT_EN, ctx=OVERDOOR)
E("ja", "en", "ui", "設定\nキャンセル\n保存", ["Settings\nCancel\nSave"], ctx="phone screen")
E("ja", "en", "label", "開封後は冷蔵庫で保存し、お早めにお召し上がりください。",
  ["After opening, keep refrigerated and consume as soon as possible."], ctx="food packaging")
E("ja", "pl", "word", "出口", "Wyjście", acc=WYJ, ctx=STATION)
E("ja", "pl", "word", "禁煙", ["Zakaz palenia"], acc=NOSMOKE_PL)
E("ja", "pl", "word", "非常口", ["Wyjście ewakuacyjne", "Wyjście awaryjne"], acc=EMERG_PL, ctx=OVERDOOR)
E("ja", "pl", "word", "準備中", ["Nieczynne", "W przygotowaniu"], acc=["nieczynn", "zamknięt", "przygotow"],
  ctx="sign hanging on a restaurant door")
E("ja", "de", "word", "非常口", ["Notausgang"], acc=["notausgang"], ctx=OVERDOOR)

# ---------------------------------------------------------------- Chinese
E("zh", "en", "word", "出口", ["Exit", "Way out"], acc=EXIT_EN, ctx=OVERDOOR)
E("zh", "en", "word", "入口", ["Entrance", "Way in"], acc=ENTRANCE_EN, ctx=OVERDOOR)
E("zh", "en", "short", "禁止吸烟", ["No smoking"], acc=NOSMOKE_EN)
E("zh", "en", "short", "小心地滑", ["Caution: wet floor", "Caution, slippery floor"], acc=["slipp", "wet floor"])
E("zh", "en", "short", "安全出口", ["Emergency exit", "Safety exit"], acc=["exit"], ctx=OVERDOOR)
E("zh", "en", "word", "推", ["Push"], acc=PUSH_EN, ctx=DOOR)
E("zh", "en", "word", "拉", ["Pull"], acc=PULL_EN, ctx=DOOR)
E("zh", "en", "short", "请勿触摸", ["Please do not touch"], acc=["not touch", "don't touch", "no touching"])
E("zh", "en", "word", "洗手间", ["Restroom", "Toilet"], acc=TOILET_EN)
E("zh", "en", "menu", "菜单\n宫保鸡丁 ￥38\n麻婆豆腐 ￥28\n扬州炒饭 ￥25",
  ["Menu\nKung Pao chicken ￥38\nMapo tofu ￥28\nYangzhou fried rice ￥25"], ctx="restaurant menu")
E("zh", "en", "time", "营业时间\n周一至周五 9:00-21:00\n周末 10:00-22:00",
  ["Business hours\nMonday to Friday 9:00-21:00\nWeekends 10:00-22:00"], ctx="shop door")
E("zh", "en", "doc", "为了您和他人的安全，请勿在车厢内饮食。",
  ["For your safety and that of others, please do not eat or drink in the carriage."], ctx="notice in a metro car")
E("zh", "en", "noise", "禁止吸姻", ["No smoking"], acc=NOSMOKE_EN)
E("zh", "en", "short", "禁止停車", ["No parking"], acc=["no parking", "parking prohibited", "parking is prohibited"])
E("zh", "en", "label", "请置于阴凉干燥处，避免阳光直射。",
  ["Store in a cool, dry place away from direct sunlight."], ctx="food packaging")
E("zh", "pl", "word", "出口", "Wyjście", acc=WYJ, ctx=OVERDOOR)
E("zh", "pl", "short", "禁止吸烟", ["Zakaz palenia"], acc=NOSMOKE_PL)
E("zh", "pl", "short", "小心台阶", ["Uwaga, stopień", "Uwaga na stopień"], acc=["stop", "schod"])
E("zh", "pl", "short", "暂停营业", ["Tymczasowo nieczynne", "Tymczasowo zamknięte"], acc=CLOSED_PL)

# ---------------------------------------------------------------- Korean
E("ko", "en", "word", "출구", ["Exit", "Way out"], acc=EXIT_EN, ctx=STATION)
E("ko", "en", "word", "입구", ["Entrance", "Way in"], acc=ENTRANCE_EN, ctx=OVERDOOR)
E("ko", "en", "word", "비상구", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("ko", "en", "word", "금연", ["No smoking"], acc=NOSMOKE_EN + ["non-smoking"])
E("ko", "en", "word", "당기세요", ["Pull"], acc=PULL_EN, ctx=DOOR)
E("ko", "en", "word", "미세요", ["Push"], acc=PUSH_EN, ctx=DOOR)
E("ko", "en", "word", "화장실", ["Restroom", "Toilet"], acc=TOILET_EN)
E("ko", "en", "word", "출입금지", ["No entry", "Keep out"], acc=NOENTRY_EN)
E("ko", "en", "menu", "메뉴\n김치찌개 8,000원\n비빔밥 9,000원\n불고기 정식 13,000원",
  ["Menu\nKimchi stew 8,000 won\nBibimbap 9,000 won\nBulgogi set meal 13,000 won"], ctx="restaurant menu")
E("ko", "en", "doc", "안전선 밖으로 물러서 주시기 바랍니다.", ["Please step back behind the safety line."],
  ctx="sign on a subway platform")
E("ko", "pl", "word", "출구", "Wyjście", acc=WYJ, ctx=STATION)
E("ko", "pl", "word", "금연", ["Zakaz palenia"], acc=NOSMOKE_PL)
E("ko", "pl", "word", "화장실", ["Toaleta", "WC"], acc=["toalet", "wc", "łazienk"])

# ---------------------------------------------------------------- Arabic
E("ar", "en", "word", "مخرج", ["Exit", "Way out"], acc=EXIT_EN, ctx=OVERDOOR)
E("ar", "en", "word", "مدخل", ["Entrance", "Way in"], acc=ENTRANCE_EN, ctx=OVERDOOR)
E("ar", "en", "short", "ممنوع التدخين", ["No smoking"], acc=NOSMOKE_EN)
E("ar", "en", "short", "مخرج طوارئ", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("ar", "en", "word", "خطر", ["Danger"], acc=["danger"])
E("ar", "en", "word", "مغلق", ["Closed"], acc=["closed"], ctx="sign on a shop door")
E("ar", "en", "warn", "الرجاء خلع الأحذية", ["Please remove your shoes", "Please take off your shoes"],
  acc=["shoes"], ctx="sign at a mosque entrance")
E("ar", "en", "menu", "قائمة الطعام\nحمص ١٥ درهم\nفلافل ١٠ دراهم\nشاورما دجاج ٢٥ درهم",
  ["Menu\nHummus 15 dirhams\nFalafel 10 dirhams\nChicken shawarma 25 dirhams"], ctx="restaurant menu")
E("ar", "en", "doc", "يمنع منعا باتا الوقوف في هذا المكان",
  ["Stopping in this place is strictly prohibited", "Parking here is strictly prohibited"],
  ctx="road sign")
E("ar", "pl", "word", "مخرج", "Wyjście", acc=WYJ, ctx=OVERDOOR)
E("ar", "pl", "short", "ممنوع التدخين", ["Zakaz palenia"], acc=NOSMOKE_PL)
E("ar", "pl", "word", "مغلق", ["Zamknięte", "Nieczynne"], acc=CLOSED_PL, ctx="sign on a shop door")

# ---------------------------------------------------------------- Turkish
E("tr", "en", "word", "Çıkış", ["Exit", "Way out"], acc=EXIT_EN, ctx=OVERDOOR)
E("tr", "en", "word", "Giriş", ["Entrance", "Way in"], acc=ENTRANCE_EN, ctx=OVERDOOR)
E("tr", "en", "short", "Sigara içilmez", ["No smoking"], acc=NOSMOKE_EN)
E("tr", "en", "caps", "ACİL ÇIKIŞ", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("tr", "en", "word", "İtiniz", ["Push"], acc=PUSH_EN, ctx=DOOR)
E("tr", "en", "word", "Çekiniz", ["Pull"], acc=PULL_EN, ctx=DOOR)
E("tr", "en", "multi", "Dikkat!\nIslak zemin", ["Caution!\nWet floor"], acc=["wet floor", "slipp"])
E("tr", "en", "menu", "Menü\nMercimek çorbası 60 TL\nAdana kebap 220 TL\nKünefe 120 TL",
  ["Menu\nLentil soup 60 TL\nAdana kebab 220 TL\nKünefe 120 TL"], ctx="restaurant menu")
E("tr", "en", "doc", "Lütfen biletinizi kontrol için hazır bulundurunuz.",
  ["Please have your ticket ready for inspection."], ctx="notice in a ferry")
E("tr", "pl", "word", "Çıkış", "Wyjście", acc=WYJ, ctx=OVERDOOR)
E("tr", "pl", "short", "Sigara içilmez", ["Zakaz palenia"], acc=NOSMOKE_PL)
E("tr", "pl", "word", "Kapalı", ["Zamknięte", "Nieczynne"], acc=CLOSED_PL, ctx="sign on a shop door")

# ---------------------------------------------------------------- Dutch
E("nl", "en", "word", "Uitgang", ["Exit", "Way out"], acc=EXIT_EN, ctx=OVERDOOR)
E("nl", "en", "word", "Ingang", ["Entrance", "Way in"], acc=ENTRANCE_EN, ctx=OVERDOOR)
E("nl", "en", "word", "Nooduitgang", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("nl", "en", "short", "Verboden te roken", ["No smoking"], acc=NOSMOKE_EN)
E("nl", "en", "word", "Duwen", ["Push"], acc=PUSH_EN, ctx=DOOR)
E("nl", "en", "word", "Trekken", ["Pull"], acc=PULL_EN, ctx=DOOR)
E("nl", "en", "short", "Let op: gladde vloer", ["Caution: slippery floor"], acc=["slipp", "wet floor"])
E("nl", "en", "menu", "Lunchkaart\nTosti ham-kaas € 5,50\nUitsmijter met spek € 9,75\nSoep van de dag € 6,25",
  ["Lunch menu\nHam and cheese toastie € 5,50\nFried eggs on bread with bacon € 9,75\nSoup of the day € 6,25",
   "Lunch menu\nToasted ham and cheese sandwich € 5,50\nUitsmijter with bacon € 9,75\nSoup of the day € 6,25"],
  ctx="café menu")
E("nl", "en", "doc", "Fietsen plaatsen tegen de gevel is verboden. Fietsen worden verwijderd.",
  ["Parking bicycles against the façade is prohibited. Bicycles will be removed."], ctx="sign on a building")
E("nl", "en", "ff", "Gift", ["Donation", "Gift"], acc=["donation", "gift", "donate"], ctx="sign on a collection box")
E("nl", "pl", "word", "Uitgang", "Wyjście", acc=WYJ, ctx=OVERDOOR)
E("nl", "pl", "short", "Verboden toegang", ["Wstęp wzbroniony", "Zakaz wstępu"], acc=NOENTRY_PL)
E("nl", "pl", "word", "Gesloten", ["Zamknięte", "Nieczynne"], acc=CLOSED_PL, ctx="sign on a shop door")

# ---------------------------------------------------------------- Czech
E("cs", "en", "word", "Východ", ["Exit", "Way out"], acc=EXIT_EN, ctx=OVERDOOR)
E("cs", "en", "word", "Vchod", ["Entrance", "Way in"], acc=ENTRANCE_EN, ctx=OVERDOOR)
E("cs", "en", "short", "Nouzový východ", ["Emergency exit", "Fire exit"], acc=EMERG_EN, ctx=OVERDOOR)
E("cs", "en", "short", "Kouření zakázáno", ["No smoking", "Smoking prohibited"], acc=NOSMOKE_EN)
E("cs", "en", "word", "Táhnout", ["Pull"], acc=PULL_EN, ctx=DOOR)
E("cs", "en", "word", "Tlačit", ["Push"], acc=PUSH_EN, ctx=DOOR)
E("cs", "en", "caps", "POZOR! NEBEZPEČÍ ÚRAZU ELEKTRICKÝM PROUDEM", ["Caution! Danger of electric shock",
  "Warning! Risk of electric shock"], acc=["electric"])
E("cs", "en", "menu", "Jídelní lístek\nSvíčková na smetaně s knedlíkem 245 Kč\nHovězí guláš, knedlík 199 Kč\n"
  "Smažený sýr, hranolky 189 Kč",
  ["Menu\nBeef sirloin in cream sauce with dumplings 245 Kč\nBeef goulash, dumplings 199 Kč\n"
   "Fried cheese, fries 189 Kč"], ctx="restaurant menu")
E("cs", "en", "doc", "Jízdenku je nutné označit ihned po nástupu do vozu.",
  ["The ticket must be validated immediately after boarding the vehicle."], ctx="notice in a tram")
E("cs", "pl", "word", "Východ", "Wyjście", acc=WYJ, ctx=OVERDOOR)
E("cs", "pl", "word", "Pozor", ["Uwaga"], acc=["uwaga"])
E("cs", "pl", "word", "Zavřeno", ["Zamknięte", "Nieczynne"], acc=CLOSED_PL, ctx="sign on a shop door")
E("cs", "pl", "short", "Vstup zakázán", ["Wstęp wzbroniony", "Zakaz wstępu"], acc=NOENTRY_PL)

# ---------------------------------------------------------------- English (to Polish / German / Spanish)
E("en", "pl", "word", "Exit", "Wyjście", acc=WYJ, ctx=OVERDOOR)
E("en", "pl", "short", "Way out", "Wyjście", acc=WYJ, ctx="sign in a London Underground station")
E("en", "pl", "short", "Mind the gap", ["Uwaga na szczelinę", "Uważaj na odstęp między pociągiem a peronem"],
  acc=["szczelin", "odstęp", "przerw", "luk"], ctx="sign on a station platform")
E("en", "pl", "short", "Staff only", ["Tylko dla personelu", "Wstęp tylko dla personelu"],
  acc=["personel", "pracownik", "obsług"])
E("en", "pl", "short", "Wet paint", ["Świeżo malowane"], acc=["malowan", "farb"])
E("en", "pl", "caps", "OUT OF ORDER", ["Nieczynne", "Awaria"], acc=["nieczynn", "awari", "nie działa", "zepsut"],
  ctx="sign on a vending machine")
E("en", "pl", "short", "Keep left", ["Trzymaj się lewej strony", "Lewa strona"], acc=["lew"])
E("en", "pl", "menu", "Breakfast\nFull English breakfast £9.50\nPorridge with honey £4.20",
  ["Śniadanie\nPełne śniadanie angielskie £9.50\nOwsianka z miodem £4.20"], ctx="café menu")
E("en", "pl", "doc", "Please do not leave bags unattended. Unattended items may be removed and destroyed.",
  ["Prosimy nie zostawiać bagażu bez opieki. Pozostawione przedmioty mogą zostać usunięte i zniszczone."],
  ctx="notice at an airport")
E("en", "de", "short", "Emergency exit", ["Notausgang"], acc=["notausgang"], ctx=OVERDOOR)
E("en", "de", "warn", "Please do not feed the animals", ["Bitte die Tiere nicht füttern",
  "Bitte füttern Sie die Tiere nicht"], acc=["füttern"])
E("en", "es", "word", "Exit", "Salida", acc=["salida"], ctx=OVERDOOR)
E("en", "es", "short", "No smoking", ["Prohibido fumar", "No fumar"], acc=["prohibido fumar", "no fumar"])

if __name__ == "__main__":
    out = Path(__file__).with_name("testset.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for item in ITEMS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    ids = [i["id"] for i in ITEMS]
    assert len(ids) == len(set(ids))
    from collections import Counter

    print(len(ITEMS), "items ->", out)
    print("targets:", Counter(i["tgt"] for i in ITEMS))
    print("sources:", Counter(i["src"] for i in ITEMS))
    print("cats:", Counter(i["cat"] for i in ITEMS))
