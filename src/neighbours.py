"""
neighbours.py
=============
Geographic adjacency between Ukrainian oblasts, restricted to the region
labels present in the volunteer dataset. Used to build spatial features:
air-raid alerts propagate between adjacent regions (a threat crossing one
oblast often precedes an alert in the next), so a region's recent neighbour
activity is a strong predictor of its own near-future alerts.

Adjacency is undirected and hand-encoded from the political map of Ukraine.
Only first-order land neighbours are included. "Kyiv City" is treated as
sharing Kyivska oblast's neighbours (it is an enclave within it).
"""

NEIGHBOURS = {
    "Kyiv City": ["Kyivska oblast", "Chernihivska oblast", "Zhytomyrska oblast",
                  "Cherkaska oblast", "Poltavska oblast"],
    "Kyivska oblast": ["Kyiv City", "Chernihivska oblast", "Zhytomyrska oblast",
                       "Cherkaska oblast", "Poltavska oblast", "Vinnytska oblast"],
    "Chernihivska oblast": ["Kyivska oblast", "Sumska oblast", "Poltavska oblast"],
    "Sumska oblast": ["Chernihivska oblast", "Poltavska oblast", "Kharkivska oblast"],
    "Kharkivska oblast": ["Sumska oblast", "Poltavska oblast", "Dnipropetrovska oblast",
                          "Donetska oblast", "Luhanska oblast"],
    "Poltavska oblast": ["Kyivska oblast", "Chernihivska oblast", "Sumska oblast",
                         "Kharkivska oblast", "Dnipropetrovska oblast",
                         "Kirovohradska oblast", "Cherkaska oblast"],
    "Cherkaska oblast": ["Kyivska oblast", "Poltavska oblast", "Kirovohradska oblast",
                         "Vinnytska oblast"],
    "Dnipropetrovska oblast": ["Kharkivska oblast", "Poltavska oblast",
                               "Kirovohradska oblast", "Mykolaivska oblast",
                               "Khersonska oblast", "Zaporizka oblast", "Donetska oblast"],
    "Donetska oblast": ["Kharkivska oblast", "Dnipropetrovska oblast",
                        "Zaporizka oblast", "Luhanska oblast"],
    "Zaporizka oblast": ["Dnipropetrovska oblast", "Donetska oblast",
                         "Khersonska oblast"],
    "Khersonska oblast": ["Dnipropetrovska oblast", "Zaporizka oblast",
                          "Mykolaivska oblast"],
    "Mykolaivska oblast": ["Khersonska oblast", "Dnipropetrovska oblast",
                           "Kirovohradska oblast", "Odeska oblast"],
    "Odeska oblast": ["Mykolaivska oblast", "Kirovohradska oblast", "Vinnytska oblast"],
    "Kirovohradska oblast": ["Cherkaska oblast", "Poltavska oblast",
                             "Dnipropetrovska oblast", "Mykolaivska oblast",
                             "Odeska oblast", "Vinnytska oblast"],
    "Vinnytska oblast": ["Kyivska oblast", "Cherkaska oblast", "Kirovohradska oblast",
                         "Odeska oblast", "Khmelnytska oblast", "Zhytomyrska oblast"],
    "Zhytomyrska oblast": ["Kyiv City", "Kyivska oblast", "Vinnytska oblast",
                           "Khmelnytska oblast", "Rivnenska oblast"],
    "Khmelnytska oblast": ["Vinnytska oblast", "Zhytomyrska oblast", "Rivnenska oblast",
                           "Ternopilska oblast", "Chernivetska oblast"],
    "Rivnenska oblast": ["Zhytomyrska oblast", "Khmelnytska oblast", "Ternopilska oblast",
                         "Lvivska oblast", "Volynska oblast"],
    "Ternopilska oblast": ["Rivnenska oblast", "Khmelnytska oblast", "Chernivetska oblast",
                           "Ivano-Frankivska oblast", "Lvivska oblast"],
    "Lvivska oblast": ["Volynska oblast", "Rivnenska oblast", "Ternopilska oblast",
                       "Ivano-Frankivska oblast", "Zakarpatska oblast"],
    "Volynska oblast": ["Lvivska oblast", "Rivnenska oblast"],
    "Ivano-Frankivska oblast": ["Lvivska oblast", "Ternopilska oblast",
                                "Chernivetska oblast", "Zakarpatska oblast"],
    "Chernivetska oblast": ["Ivano-Frankivska oblast", "Ternopilska oblast",
                            "Khmelnytska oblast"],
    "Zakarpatska oblast": ["Lvivska oblast", "Ivano-Frankivska oblast"],
}


def get_neighbours(region: str) -> list[str]:
    if region not in NEIGHBOURS:
        raise ValueError(f"No adjacency defined for {region!r}.")
    return NEIGHBOURS[region]
