# NFL teams data for Python backend
NFL_TEAMS = [
    # AFC East
    {"id": "buf", "name": "Bills", "city": "Buffalo", "abbreviation": "BUF", "conference": "AFC", "division": "East", "primary_color": "#00338D", "secondary_color": "#C60C30"},
    {"id": "mia", "name": "Dolphins", "city": "Miami", "abbreviation": "MIA", "conference": "AFC", "division": "East", "primary_color": "#008E97", "secondary_color": "#FC4C02"},
    {"id": "ne", "name": "Patriots", "city": "New England", "abbreviation": "NE", "conference": "AFC", "division": "East", "primary_color": "#002244", "secondary_color": "#C60C30"},
    {"id": "nyj", "name": "Jets", "city": "New York", "abbreviation": "NYJ", "conference": "AFC", "division": "East", "primary_color": "#125740", "secondary_color": "#FFFFFF"},

    # AFC North
    {"id": "bal", "name": "Ravens", "city": "Baltimore", "abbreviation": "BAL", "conference": "AFC", "division": "North", "primary_color": "#241773", "secondary_color": "#000000"},
    {"id": "cin", "name": "Bengals", "city": "Cincinnati", "abbreviation": "CIN", "conference": "AFC", "division": "North", "primary_color": "#FB4F14", "secondary_color": "#000000"},
    {"id": "cle", "name": "Browns", "city": "Cleveland", "abbreviation": "CLE", "conference": "AFC", "division": "North", "primary_color": "#311D00", "secondary_color": "#FF3C00"},
    {"id": "pit", "name": "Steelers", "city": "Pittsburgh", "abbreviation": "PIT", "conference": "AFC", "division": "North", "primary_color": "#FFB612", "secondary_color": "#000000"},

    # AFC South
    {"id": "hou", "name": "Texans", "city": "Houston", "abbreviation": "HOU", "conference": "AFC", "division": "South", "primary_color": "#03202F", "secondary_color": "#A71930"},
    {"id": "ind", "name": "Colts", "city": "Indianapolis", "abbreviation": "IND", "conference": "AFC", "division": "South", "primary_color": "#002C5F", "secondary_color": "#A2AAAD"},
    {"id": "jax", "name": "Jaguars", "city": "Jacksonville", "abbreviation": "JAX", "conference": "AFC", "division": "South", "primary_color": "#006778", "secondary_color": "#D7A22A"},
    {"id": "ten", "name": "Titans", "city": "Tennessee", "abbreviation": "TEN", "conference": "AFC", "division": "South", "primary_color": "#0C2340", "secondary_color": "#4B92DB"},

    # AFC West
    {"id": "den", "name": "Broncos", "city": "Denver", "abbreviation": "DEN", "conference": "AFC", "division": "West", "primary_color": "#FB4F14", "secondary_color": "#002244"},
    {"id": "kc", "name": "Chiefs", "city": "Kansas City", "abbreviation": "KC", "conference": "AFC", "division": "West", "primary_color": "#E31837", "secondary_color": "#FFB81C"},
    {"id": "lv", "name": "Raiders", "city": "Las Vegas", "abbreviation": "LV", "conference": "AFC", "division": "West", "primary_color": "#000000", "secondary_color": "#A5ACAF"},
    {"id": "lac", "name": "Chargers", "city": "Los Angeles", "abbreviation": "LAC", "conference": "AFC", "division": "West", "primary_color": "#0080C6", "secondary_color": "#FFC20E"},

    # NFC East
    {"id": "dal", "name": "Cowboys", "city": "Dallas", "abbreviation": "DAL", "conference": "NFC", "division": "East", "primary_color": "#003594", "secondary_color": "#041E42"},
    {"id": "nyg", "name": "Giants", "city": "New York", "abbreviation": "NYG", "conference": "NFC", "division": "East", "primary_color": "#0B2265", "secondary_color": "#A71930"},
    {"id": "phi", "name": "Eagles", "city": "Philadelphia", "abbreviation": "PHI", "conference": "NFC", "division": "East", "primary_color": "#004C54", "secondary_color": "#A5ACAF"},
    {"id": "was", "name": "Commanders", "city": "Washington", "abbreviation": "WAS", "conference": "NFC", "division": "East", "primary_color": "#5A1414", "secondary_color": "#FFB612"},

    # NFC North
    {"id": "chi", "name": "Bears", "city": "Chicago", "abbreviation": "CHI", "conference": "NFC", "division": "North", "primary_color": "#0B162A", "secondary_color": "#C83803"},
    {"id": "det", "name": "Lions", "city": "Detroit", "abbreviation": "DET", "conference": "NFC", "division": "North", "primary_color": "#0076B6", "secondary_color": "#B0B7BC"},
    {"id": "gb", "name": "Packers", "city": "Green Bay", "abbreviation": "GB", "conference": "NFC", "division": "North", "primary_color": "#203731", "secondary_color": "#FFB612"},
    {"id": "min", "name": "Vikings", "city": "Minnesota", "abbreviation": "MIN", "conference": "NFC", "division": "North", "primary_color": "#4F2683", "secondary_color": "#FFC62F"},

    # NFC South
    {"id": "atl", "name": "Falcons", "city": "Atlanta", "abbreviation": "ATL", "conference": "NFC", "division": "South", "primary_color": "#A71930", "secondary_color": "#000000"},
    {"id": "car", "name": "Panthers", "city": "Carolina", "abbreviation": "CAR", "conference": "NFC", "division": "South", "primary_color": "#0085CA", "secondary_color": "#101820"},
    {"id": "no", "name": "Saints", "city": "New Orleans", "abbreviation": "NO", "conference": "NFC", "division": "South", "primary_color": "#D3BC8D", "secondary_color": "#101820"},
    {"id": "tb", "name": "Buccaneers", "city": "Tampa Bay", "abbreviation": "TB", "conference": "NFC", "division": "South", "primary_color": "#D50A0A", "secondary_color": "#FF7900"},

    # NFC West
    {"id": "ari", "name": "Cardinals", "city": "Arizona", "abbreviation": "ARI", "conference": "NFC", "division": "West", "primary_color": "#97233F", "secondary_color": "#000000"},
    {"id": "lar", "name": "Rams", "city": "Los Angeles", "abbreviation": "LAR", "conference": "NFC", "division": "West", "primary_color": "#003594", "secondary_color": "#FFA300"},
    {"id": "sf", "name": "49ers", "city": "San Francisco", "abbreviation": "SF", "conference": "NFC", "division": "West", "primary_color": "#AA0000", "secondary_color": "#B3995D"},
    {"id": "sea", "name": "Seahawks", "city": "Seattle", "abbreviation": "SEA", "conference": "NFC", "division": "West", "primary_color": "#002244", "secondary_color": "#69BE28"},
]