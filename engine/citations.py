"""
Potatostics Academic Citation Engine.
Provides programmatic access to peer-reviewed citations, DOIs, and BibTeX entries
for all econometric and statistical routines implemented in Potatostics.
"""

from typing import Dict, List, Optional


CITATIONS_DB: Dict[str, List[Dict[str, str]]] = {
    "regress": [
        {
            "key": "white1980heteroskedasticity",
            "author": "White, Halbert",
            "year": "1980",
            "title": "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity",
            "journal": "Econometrica",
            "volume": "48",
            "issue": "4",
            "pages": "817-838",
            "doi": "10.2307/1912934",
            "bibtex": """@article{white1980heteroskedasticity,
  author    = {White, Halbert},
  title     = {A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity},
  journal   = {Econometrica},
  year      = {1980},
  volume    = {48},
  number    = {4},
  pages     = {817--838},
  doi       = {10.2307/1912934}
}"""
        },
        {
            "key": "mackinnon1985some",
            "author": "MacKinnon, James G., & White, Halbert",
            "year": "1985",
            "title": "Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties",
            "journal": "Journal of Econometrics",
            "volume": "29",
            "issue": "3",
            "pages": "305-325",
            "doi": "10.1016/0304-4076(85)90158-7",
            "bibtex": """@article{mackinnon1985some,
  author    = {MacKinnon, James G. and White, Halbert},
  title     = {Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties},
  journal   = {Journal of Econometrics},
  year      = {1985},
  volume    = {29},
  number    = {3},
  pages     = {305--325},
  doi       = {10.1016/0304-4076(85)90158-7}
}"""
        }
    ],
    "cluster": [
        {
            "key": "liang1986longitudinal",
            "author": "Liang, Kung-Yee, & Zeger, Scott L.",
            "year": "1986",
            "title": "Longitudinal data analysis using generalized linear models",
            "journal": "Biometrika",
            "volume": "73",
            "issue": "1",
            "pages": "13-22",
            "doi": "10.1093/biomet/73.1.13",
            "bibtex": """@article{liang1986longitudinal,
  author    = {Liang, Kung-Yee and Zeger, Scott L.},
  title     = {Longitudinal data analysis using generalized linear models},
  journal   = {Biometrika},
  year      = {1986},
  volume    = {73},
  number    = {1},
  pages     = {13--22},
  doi       = {10.1093/biomet/73.1.13}
}"""
        },
        {
            "key": "arellano1987computing",
            "author": "Arellano, Manuel",
            "year": "1987",
            "title": "Computing Robust Standard Errors for Models with Fixed Effects",
            "journal": "Oxford Bulletin of Economics and Statistics",
            "volume": "49",
            "issue": "4",
            "pages": "431-434",
            "doi": "10.1111/j.1468-0084.1987.mp49004006.x",
            "bibtex": """@article{arellano1987computing,
  author    = {Arellano, Manuel},
  title     = {Computing Robust Standard Errors for Models with Fixed Effects},
  journal   = {Oxford Bulletin of Economics and Statistics},
  year      = {1987},
  volume    = {49},
  number    = {4},
  pages     = {431--434},
  doi       = {10.1111/j.1468-0084.1987.mp49004006.x}
}"""
        }
    ],
    "xtreg": [
        {
            "key": "swamy1972exact",
            "author": "Swamy, P. A. V. B., & Arora, S. S.",
            "year": "1972",
            "title": "The Exact Finite Sample Properties of the Estimators of Coefficients in the Error Components Regression Models",
            "journal": "Econometrica",
            "volume": "40",
            "issue": "2",
            "pages": "261-275",
            "doi": "10.2307/1913165",
            "bibtex": """@article{swamy1972exact,
  author    = {Swamy, P. A. V. B. and Arora, S. S.},
  title     = {The Exact Finite Sample Properties of the Estimators of Coefficients in the Error Components Regression Models},
  journal   = {Econometrica},
  year      = {1972},
  volume    = {40},
  number    = {2},
  pages     = {261--275},
  doi       = {10.2307/1913165}
}"""
        },
        {
            "key": "mundlak1978pooling",
            "author": "Mundlak, Yair",
            "year": "1978",
            "title": "On the Pooling of Time Series and Cross Section Data",
            "journal": "Econometrica",
            "volume": "46",
            "issue": "1",
            "pages": "69-85",
            "doi": "10.2307/1913646",
            "bibtex": """@article{mundlak1978pooling,
  author    = {Mundlak, Yair},
  title     = {On the Pooling of Time Series and Cross Section Data},
  journal   = {Econometrica},
  year      = {1978},
  volume    = {46},
  number    = {1},
  pages     = {69--85},
  doi       = {10.2307/1913646}
}"""
        }
    ],
    "hausman": [
        {
            "key": "hausman1978specification",
            "author": "Hausman, Jerry A.",
            "year": "1978",
            "title": "Specification Tests in Econometrics",
            "journal": "Econometrica",
            "volume": "46",
            "issue": "6",
            "pages": "1251-1271",
            "doi": "10.2307/1913827",
            "bibtex": """@article{hausman1978specification,
  author    = {Hausman, Jerry A.},
  title     = {Specification Tests in Econometrics},
  journal   = {Econometrica},
  year      = {1978},
  volume    = {46},
  number    = {6},
  pages     = {1251--1271},
  doi       = {10.2307/1913827}
}"""
        }
    ],
    "reghdfe": [
        {
            "key": "guimaraes2010simple",
            "author": "Guimarães, Paulo, & Portugal, Pedro",
            "year": "2010",
            "title": "A simple feasible alternative procedure to estimate models with high-dimensional fixed effects",
            "journal": "Econometric Software Journal",
            "volume": "10",
            "issue": "4",
            "pages": "628-649",
            "doi": "10.1177/1536867X1001000406",
            "bibtex": r"""@article{guimaraes2010simple,
  author    = {Guimar{\~a}es, Paulo and Portugal, Pedro},
  title     = {A simple feasible alternative procedure to estimate models with high-dimensional fixed effects},
  journal   = {Econometric Software Journal},
  year      = {2010},
  volume    = {10},
  number    = {4},
  pages     = {628--649},
  doi       = {10.1177/1536867X1001000406}
}"""
        }
    ],
    "ivregress": [
        {
            "key": "basmann1957generalized",
            "author": "Basmann, R. L.",
            "year": "1957",
            "title": "A Generalized Classical Method of Linear Estimation of Coefficients in a Structural Equation",
            "journal": "Econometrica",
            "volume": "25",
            "issue": "1",
            "pages": "77-83",
            "doi": "10.2307/1907743",
            "bibtex": """@article{basmann1957generalized,
  author    = {Basmann, R. L.},
  title     = {A Generalized Classical Method of Linear Estimation of Coefficients in a Structural Equation},
  journal   = {Econometrica},
  year      = {1957},
  volume    = {25},
  number    = {1},
  pages     = {77--83},
  doi       = {10.2307/1907743}
}"""
        },
        {
            "key": "sargan1958estimation",
            "author": "Sargan, J. D.",
            "year": "1958",
            "title": "The Estimation of Economic Relationships using Instrumental Variables",
            "journal": "Econometrica",
            "volume": "26",
            "issue": "3",
            "pages": "393-415",
            "doi": "10.2307/1907619",
            "bibtex": """@article{sargan1958estimation,
  author    = {Sargan, J. D.},
  title     = {The Estimation of Economic Relationships using Instrumental Variables},
  journal   = {Econometrica},
  year      = {1958},
  volume    = {26},
  number    = {3},
  pages     = {393--415},
  doi       = {10.2307/1907619}
}"""
        }
    ],
    "weakiv": [
        {
            "key": "montiel2013robust",
            "author": "Montiel Olea, José Luis, & Pflueger, Carolin",
            "year": "2013",
            "title": "A Robust Test for Weak Instruments",
            "journal": "Journal of Business & Economic Statistics",
            "volume": "31",
            "issue": "3",
            "pages": "358-369",
            "doi": "10.1080/07350015.2013.804173",
            "bibtex": r"""@article{montiel2013robust,
  author    = {Montiel Olea, Jos{\'e} Luis and Pflueger, Carolin},
  title     = {A Robust Test for Weak Instruments},
  journal   = {Journal of Business \& Economic Statistics},
  year      = {2013},
  volume    = {31},
  number    = {3},
  pages     = {358--369},
  doi       = {10.1080/07350015.2013.804173}
}"""
        },
        {
            "key": "stock2005testing",
            "author": "Stock, James H., & Yogo, Motohiro",
            "year": "2005",
            "title": "Testing for Weak Instruments in Linear IV Regression",
            "journal": "Cambridge University Press",
            "pages": "80-108",
            "doi": "10.1017/CBO9780511614491.006",
            "bibtex": """@incollection{stock2005testing,
  author    = {Stock, James H. and Yogo, Motohiro},
  title     = {Testing for Weak Instruments in Linear IV Regression},
  booktitle = {Identification and Inference for Econometric Models},
  publisher = {Cambridge University Press},
  year      = {2005},
  pages     = {80--108},
  doi       = {10.1017/CBO9780511614491.006}
}"""
        }
    ],
    "csdid": [
        {
            "key": "callaway2021difference",
            "author": "Callaway, Brantly, & Sant'Anna, Pedro H. C.",
            "year": "2021",
            "title": "Difference-in-Differences with multiple time periods",
            "journal": "Journal of Econometrics",
            "volume": "225",
            "issue": "2",
            "pages": "200-230",
            "doi": "10.1016/j.jeconom.2020.12.001",
            "bibtex": """@article{callaway2021difference,
  author    = {Callaway, Brantly and Sant'Anna, Pedro H. C.},
  title     = {Difference-in-Differences with multiple time periods},
  journal   = {Journal of Econometrics},
  year      = {2021},
  volume    = {225},
  number    = {2},
  pages     = {200--230},
  doi       = {10.1016/j.jeconom.2020.12.001}
}"""
        }
    ],
    "synth": [
        {
            "key": "abadie2003economic",
            "author": "Abadie, Alberto, & Gardeazabal, Javier",
            "year": "2003",
            "title": "The Economic Costs of Conflict: A Case Study of the Basque Country",
            "journal": "American Economic Review",
            "volume": "93",
            "issue": "1",
            "pages": "113-132",
            "doi": "10.1257/000282803321455188",
            "bibtex": """@article{abadie2003economic,
  author    = {Abadie, Alberto and Gardeazabal, Javier},
  title     = {The Economic Costs of Conflict: A Case Study of the Basque Country},
  journal   = {American Economic Review},
  year      = {2003},
  volume    = {93},
  number    = {1},
  pages     = {113--132},
  doi       = {10.1257/000282803321455188}
}"""
        },
        {
            "key": "abadie2010synthetic",
            "author": "Abadie, Alberto, Diamond, Alexis, & Hainmueller, Jens",
            "year": "2010",
            "title": "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program",
            "journal": "Journal of the American Statistical Association",
            "volume": "105",
            "issue": "490",
            "pages": "493-505",
            "doi": "10.1198/jasa.2009.ap08746",
            "bibtex": """@article{abadie2010synthetic,
  author    = {Abadie, Alberto and Diamond, Alexis and Hainmueller, Jens},
  title     = {Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program},
  journal   = {Journal of the American Statistical Association},
  year      = {2010},
  volume    = {105},
  number    = {490},
  pages     = {493--505},
  doi       = {10.1198/jasa.2009.ap08746}
}"""
        }
    ],
    "eventstudy": [
        {
            "key": "sun2021estimating",
            "author": "Sun, Liyang, & Abraham, Sarah",
            "year": "2021",
            "title": "Estimating dynamic treatment effects in event studies with heterogeneous treatment effects",
            "journal": "Journal of Econometrics",
            "volume": "225",
            "issue": "2",
            "pages": "175-199",
            "doi": "10.1016/j.jeconom.2020.09.006",
            "bibtex": """@article{sun2021estimating,
  author    = {Sun, Liyang and Abraham, Sarah},
  title     = {Estimating dynamic treatment effects in event studies with heterogeneous treatment effects},
  journal   = {Journal of Econometrics},
  year      = {2021},
  volume    = {225},
  number    = {2},
  pages     = {175--199},
  doi       = {10.1016/j.jeconom.2020.09.006}
}"""
        }
    ],
    "did_multiplegt": [
        {
            "key": "dechaisemartin2020twoway",
            "author": "de Chaisemartin, Clément, & D'Haultfœuille, Xavier",
            "year": "2020",
            "title": "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects",
            "journal": "American Economic Review",
            "volume": "110",
            "issue": "9",
            "pages": "2964-2996",
            "doi": "10.1257/aer.20181169",
            "bibtex": r"""@article{dechaisemartin2020twoway,
  author    = {de Chaisemartin, Cl{\'e}ment and D'Haultf{\oe}uille, Xavier},
  title     = {Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects},
  journal   = {American Economic Review},
  year      = {2020},
  volume    = {110},
  number    = {9},
  pages     = {2964--2996},
  doi       = {10.1257/aer.20181169}
}"""
        }
    ],
    "bacondecomp": [
        {
            "key": "goodmanbacon2021difference",
            "author": "Goodman-Bacon, Andrew",
            "year": "2021",
            "title": "Difference-in-differences with variation in treatment timing",
            "journal": "Journal of Econometrics",
            "volume": "225",
            "issue": "2",
            "pages": "254-277",
            "doi": "10.1016/j.jeconom.2021.03.014",
            "bibtex": """@article{goodmanbacon2021difference,
  author    = {Goodman-Bacon, Andrew},
  title     = {Difference-in-differences with variation in treatment timing},
  journal   = {Journal of Econometrics},
  year      = {2021},
  volume    = {225},
  number    = {2},
  pages     = {254--277},
  doi       = {10.1016/j.jeconom.2021.03.014}
}"""
        }
    ],
    "psmatch": [
        {
            "key": "rosenbaum1983central",
            "author": "Rosenbaum, Paul R., & Rubin, Donald B.",
            "year": "1983",
            "title": "The central role of the propensity score in observational studies for causal effects",
            "journal": "Biometrika",
            "volume": "70",
            "issue": "1",
            "pages": "41-55",
            "doi": "10.1093/biomet/70.1.41",
            "bibtex": """@article{rosenbaum1983central,
  author    = {Rosenbaum, Paul R. and Rubin, Donald B.},
  title     = {The central role of the propensity score in observational studies for causal effects},
  journal   = {Biometrika},
  year      = {1983},
  volume    = {70},
  number    = {1},
  pages     = {41--55},
  doi       = {10.1093/biomet/70.1.41}
}"""
        }
    ],
    "tobit": [
        {
            "key": "tobin1958estimation",
            "author": "Tobin, James",
            "year": "1958",
            "title": "Estimation of Relationships for Generalized Dependent Variables",
            "journal": "Econometrica",
            "volume": "26",
            "issue": "1",
            "pages": "24-36",
            "doi": "10.2307/1907382",
            "bibtex": """@article{tobin1958estimation,
  author    = {Tobin, James},
  title     = {Estimation of Relationships for Generalized Dependent Variables},
  journal   = {Econometrica},
  year      = {1958},
  volume    = {26},
  number    = {1},
  pages     = {24--36},
  doi       = {10.2307/1907382}
}"""
        }
    ],
    "qreg": [
        {
            "key": "koenker1978regression",
            "author": "Koenker, Roger, & Bassett, Gilbert",
            "year": "1978",
            "title": "Regression Quantiles",
            "journal": "Econometrica",
            "volume": "46",
            "issue": "1",
            "pages": "33-50",
            "doi": "10.2307/1913643",
            "bibtex": """@article{koenker1978regression,
  author    = {Koenker, Roger and Bassett, Gilbert},
  title     = {Regression Quantiles},
  journal   = {Econometrica},
  year      = {1978},
  volume    = {46},
  number    = {1},
  pages     = {33--50},
  doi       = {10.2307/1913643}
}"""
        }
    ],
    "poisson": [
        {
            "key": "cameron1986econometric",
            "author": "Cameron, A. Colin, & Trivedi, Pravin K.",
            "year": "1986",
            "title": "Econometric Models Based on Count Data: Comparisons and Applications of Some Estimators and Tests",
            "journal": "Journal of Applied Econometrics",
            "volume": "1",
            "issue": "1",
            "pages": "29-53",
            "doi": "10.1002/jae.3950010104",
            "bibtex": """@article{cameron1986econometric,
  author    = {Cameron, A. Colin and Trivedi, Pravin K.},
  title     = {Econometric Models Based on Count Data: Comparisons and Applications of Some Estimators and Tests},
  journal   = {Journal of Applied Econometrics},
  year      = {1986},
  volume    = {1},
  number    = {1},
  pages     = {29--53},
  doi       = {10.1002/jae.3950010104}
}"""
        }
    ],
    "nbreg": [
        {
            "key": "cameron1986econometric",
            "author": "Cameron, A. Colin, & Trivedi, Pravin K.",
            "year": "1986",
            "title": "Econometric Models Based on Count Data: Comparisons and Applications of Some Estimators and Tests",
            "journal": "Journal of Applied Econometrics",
            "volume": "1",
            "issue": "1",
            "pages": "29-53",
            "doi": "10.1002/jae.3950010104",
            "bibtex": """@article{cameron1986econometric,
  author    = {Cameron, A. Colin and Trivedi, Pravin K.},
  title     = {Econometric Models Based on Count Data: Comparisons and Applications of Some Estimators and Tests},
  journal   = {Journal of Applied Econometrics},
  year      = {1986},
  volume    = {1},
  number    = {1},
  pages     = {29--53},
  doi       = {10.1002/jae.3950010104}
}"""
        }
    ],
    "brant": [
        {
            "key": "brant1990assessing",
            "author": "Brant, Rollin",
            "year": "1990",
            "title": "Assessing Proportionality in the Proportional Odds Model for Ordinal Logistic Regression",
            "journal": "Biometrics",
            "volume": "46",
            "issue": "4",
            "pages": "1171-1178",
            "doi": "10.2307/2532104",
            "bibtex": """@article{brant1990assessing,
  author    = {Brant, Rollin},
  title     = {Assessing Proportionality in the Proportional Odds Model for Ordinal Logistic Regression},
  journal   = {Biometrics},
  year      = {1990},
  volume    = {46},
  number    = {4},
  pages     = {1171--1178},
  doi       = {10.2307/2532104}
}"""
        }
    ],
    "sureg": [
        {
            "key": "zellner1962efficient",
            "author": "Zellner, Arnold",
            "year": "1962",
            "title": "An Efficient Method of Estimating Seemingly Unrelated Regressions and Tests for Aggregation Bias",
            "journal": "Journal of the American Statistical Association",
            "volume": "57",
            "issue": "298",
            "pages": "348-368",
            "doi": "10.1080/01621459.1962.10482864",
            "bibtex": """@article{zellner1962efficient,
  author    = {Zellner, Arnold},
  title     = {An Efficient Method of Estimating Seemingly Unrelated Regressions and Tests for Aggregation Bias},
  journal   = {Journal of the American Statistical Association},
  year      = {1962},
  volume    = {57},
  number    = {298},
  pages     = {348--368},
  doi       = {10.1080/01621459.1962.10482864}
}"""
        }
    ],
    "cnsreg": [
        {
            "key": "theil1971principles",
            "author": "Theil, Henri",
            "year": "1971",
            "title": "Principles of Econometrics",
            "journal": "John Wiley & Sons, New York",
            "doi": "ISBN: 0-471-85845-5",
            "bibtex": r"""@book{theil1971principles,
  author    = {Theil, Henri},
  title     = {Principles of Econometrics},
  publisher = {John Wiley \& Sons},
  year      = {1971},
  address   = {New York},
  isbn      = {0-471-85845-5}
}"""
        }
    ],
    "nlcom": [
        {
            "key": "rao1973linear",
            "author": "Rao, C. Radhakrishna",
            "year": "1973",
            "title": "Linear Statistical Inference and Its Applications",
            "journal": "John Wiley & Sons",
            "doi": "10.1002/9780470316436",
            "bibtex": r"""@book{rao1973linear,
  author    = {Rao, C. Radhakrishna},
  title     = {Linear Statistical Inference and Its Applications},
  edition   = {2nd},
  publisher = {John Wiley \& Sons},
  year      = {1973},
  doi       = {10.1002/9780470316436}
}"""
        }
    ],
    "testnl": [
        {
            "key": "rao1973linear",
            "author": "Rao, C. Radhakrishna",
            "year": "1973",
            "title": "Linear Statistical Inference and Its Applications",
            "journal": "John Wiley & Sons",
            "doi": "10.1002/9780470316436",
            "bibtex": r"""@book{rao1973linear,
  author    = {Rao, C. Radhakrishna},
  title     = {Linear Statistical Inference and Its Applications},
  edition   = {2nd},
  publisher = {John Wiley \& Sons},
  year      = {1973},
  doi       = {10.1002/9780470316436}
}"""
        }
    ],
    "tsfilter": [
        {
            "key": "hodrick1997postwar",
            "author": "Hodrick, Robert J., & Prescott, Edward C.",
            "year": "1997",
            "title": "Postwar U.S. Business Cycles: An Empirical Investigation",
            "journal": "Journal of Money, Credit and Banking",
            "volume": "29",
            "issue": "1",
            "pages": "1-16",
            "doi": "10.2307/2953682",
            "bibtex": """@article{hodrick1997postwar,
  author    = {Hodrick, Robert J. and Prescott, Edward C.},
  title     = {Postwar U.S. Business Cycles: An Empirical Investigation},
  journal   = {Journal of Money, Credit and Banking},
  year      = {1997},
  volume    = {29},
  number    = {1},
  pages     = {1--16},
  doi       = {10.2307/2953682}
}"""
        }
    ],
    "var": [
        {
            "key": "sims1980macroeconomics",
            "author": "Sims, Christopher A.",
            "year": "1980",
            "title": "Macroeconomics and Reality",
            "journal": "Econometrica",
            "volume": "48",
            "issue": "1",
            "pages": "1-48",
            "doi": "10.2307/1912017",
            "bibtex": """@article{sims1980macroeconomics,
  author    = {Sims, Christopher A.},
  title     = {Macroeconomics and Reality},
  journal   = {Econometrica},
  year      = {1980},
  volume    = {48},
  number    = {1},
  pages     = {1--48},
  doi       = {10.2307/1912017}
}"""
        },
        {
            "key": "lutkepohl2005new",
            "author": "Lütkepohl, Helmut",
            "year": "2005",
            "title": "New Introduction to Multiple Time Series Analysis",
            "journal": "Springer, Berlin, Heidelberg",
            "doi": "10.1007/978-3-540-27752-1",
            "bibtex": """@book{lutkepohl2005new,
  author    = {L{\"u}tkepohl, Helmut},
  title     = {New Introduction to Multiple Time Series Analysis},
  publisher = {Springer},
  year      = {2005},
  address   = {Berlin, Heidelberg},
  doi       = {10.1007/978-3-540-27752-1}
}"""
        }
    ],
    "svy": [
        {
            "key": "cochran1977sampling",
            "author": "Cochran, William G.",
            "year": "1977",
            "title": "Sampling Techniques (3rd ed.)",
            "journal": "John Wiley & Sons, New York",
            "doi": "ISBN: 0-471-16240-X",
            "bibtex": r"""@book{cochran1977sampling,
  author    = {Cochran, William G.},
  title     = {Sampling Techniques},
  edition   = {3rd},
  publisher = {John Wiley \& Sons},
  year      = {1977},
  address   = {New York},
  isbn      = {0-471-16240-X}
}"""
        },
        {
            "key": "lumley2004analysis",
            "author": "Lumley, Thomas",
            "year": "2004",
            "title": "Analysis of Complex Survey Samples",
            "journal": "Journal of Statistical Software",
            "volume": "9",
            "issue": "8",
            "pages": "1-19",
            "doi": "10.18637/jss.v009.i08",
            "bibtex": """@article{lumley2004analysis,
  author    = {Lumley, Thomas},
  title     = {Analysis of Complex Survey Samples},
  journal   = {Journal of Statistical Software},
  year      = {2004},
  volume    = {9},
  number    = {8},
  pages     = {1--19},
  doi       = {10.18637/jss.v009.i08}
}"""
        }
    ],
    "sts": [
        {
            "key": "kaplan1958nonparametric",
            "author": "Kaplan, E. L., & Meier, Paul",
            "year": "1958",
            "title": "Nonparametric Estimation from Incomplete Observations",
            "journal": "Journal of the American Statistical Association",
            "volume": "53",
            "issue": "282",
            "pages": "457-481",
            "doi": "10.1080/01621459.1958.10501452",
            "bibtex": """@article{kaplan1958nonparametric,
  author    = {Kaplan, E. L. and Meier, Paul},
  title     = {Nonparametric Estimation from Incomplete Observations},
  journal   = {Journal of the American Statistical Association},
  year      = {1958},
  volume    = {53},
  number    = {282},
  pages     = {457--481},
  doi       = {10.1080/01621459.1958.10501452}
}"""
        }
    ],
    "stcox": [
        {
            "key": "cox1972regression",
            "author": "Cox, D. R.",
            "year": "1972",
            "title": "Regression Models and Life-Tables",
            "journal": "Journal of the Royal Statistical Society: Series B (Methodological)",
            "volume": "34",
            "issue": "2",
            "pages": "187-202",
            "doi": "10.1111/j.2517-6161.1972.tb00899.x",
            "bibtex": """@article{cox1972regression,
  author    = {Cox, D. R.},
  title     = {Regression Models and Life-Tables},
  journal   = {Journal of the Royal Statistical Society: Series B (Methodological)},
  year      = {1972},
  volume    = {34},
  number    = {2},
  pages     = {187--202},
  doi       = {10.1111/j.2517-6161.1972.tb00899.x}
}"""
        },
        {
            "key": "breslow1974covariance",
            "author": "Breslow, N. E.",
            "year": "1974",
            "title": "Covariance Analysis of Censored Survival Data",
            "journal": "Biometrics",
            "volume": "30",
            "issue": "1",
            "pages": "89-99",
            "doi": "10.2307/2529620",
            "bibtex": """@article{breslow1974covariance,
  author    = {Breslow, N. E.},
  title     = {Covariance Analysis of Censored Survival Data},
  journal   = {Biometrics},
  year      = {1974},
  volume    = {30},
  number    = {1},
  pages     = {89--99},
  doi       = {10.2307/2529620}
}"""
        },
        {
            "key": "efron1977efficiency",
            "author": "Efron, Bradley",
            "year": "1977",
            "title": "The Efficiency of Cox's Likelihood Function for Censored Data",
            "journal": "Journal of the American Statistical Association",
            "volume": "72",
            "issue": "359",
            "pages": "557-565",
            "doi": "10.1080/01621459.1977.10479958",
            "bibtex": """@article{efron1977efficiency,
  author    = {Efron, Bradley},
  title     = {The Efficiency of Cox's Likelihood Function for Censored Data},
  journal   = {Journal of the American Statistical Association},
  year      = {1977},
  volume    = {72},
  number    = {359},
  pages     = {557--565},
  doi       = {10.1080/01621459.1977.10479958}
}"""
        }
    ],
    "hettest": [
        {
            "key": "breusch1979simple",
            "author": "Breusch, T. S., & Pagan, A. R.",
            "year": "1979",
            "title": "A Simple Test for Heteroscedasticity and Random Coefficient Variation",
            "journal": "Econometrica",
            "volume": "47",
            "issue": "5",
            "pages": "1287-1294",
            "doi": "10.2307/1911963",
            "bibtex": """@article{breusch1979simple,
  author    = {Breusch, T. S. and Pagan, A. R.},
  title     = {A Simple Test for Heteroscedasticity and Random Coefficient Variation},
  journal   = {Econometrica},
  year      = {1979},
  volume    = {47},
  number    = {5},
  pages     = {1287--1294},
  doi       = {10.2307/1911963}
}"""
        },
        {
            "key": "cook1983diagnostics",
            "author": "Cook, R. Dennis, & Weisberg, Sanford",
            "year": "1983",
            "title": "Diagnostics for Heteroscedasticity in Regression",
            "journal": "Biometrika",
            "volume": "70",
            "issue": "1",
            "pages": "1-10",
            "doi": "10.1093/biomet/70.1.1",
            "bibtex": """@article{cook1983diagnostics,
  author    = {Cook, R. Dennis and Weisberg, Sanford},
  title     = {Diagnostics for Heteroscedasticity in Regression},
  journal   = {Biometrika},
  year      = {1983},
  volume    = {70},
  number    = {1},
  pages     = {1--10},
  doi       = {10.1093/biomet/70.1.1}
}"""
        }
    ],
    "ovtest": [
        {
            "key": "ramsey1969tests",
            "author": "Ramsey, J. B.",
            "year": "1969",
            "title": "Tests for Specification Errors in Classical Linear Least-Squares Regression Analysis",
            "journal": "Journal of the Royal Statistical Society: Series B (Methodological)",
            "volume": "31",
            "issue": "2",
            "pages": "350-371",
            "doi": "10.1111/j.2517-6161.1969.tb00796.x",
            "bibtex": """@article{ramsey1969tests,
  author    = {Ramsey, J. B.},
  title     = {Tests for Specification Errors in Classical Linear Least-Squares Regression Analysis},
  journal   = {Journal of the Royal Statistical Society: Series B (Methodological)},
  year      = {1969},
  volume    = {31},
  number    = {2},
  pages     = {350--371},
  doi       = {10.1111/j.2517-6161.1969.tb00796.x}
}"""
        }
    ],
    "dwatson": [
        {
            "key": "durbin1950testing",
            "author": "Durbin, J., & Watson, G. S.",
            "year": "1950",
            "title": "Testing for Serial Correlation in Least Squares Regression: I",
            "journal": "Biometrika",
            "volume": "37",
            "issue": "3/4",
            "pages": "409-428",
            "doi": "10.1093/biomet/37.3-4.409",
            "bibtex": """@article{durbin1950testing,
  author    = {Durbin, J. and Watson, G. S.},
  title     = {Testing for Serial Correlation in Least Squares Regression: I},
  journal   = {Biometrika},
  year      = {1950},
  volume    = {37},
  number    = {3/4},
  pages     = {409--428},
  doi       = {10.1093/biomet/37.3-4.409}
}"""
        }
    ],
    "margins": [
        {
            "key": "williams2012using",
            "author": "Williams, Richard",
            "year": "2012",
            "title": "Using the margins command to estimate and interpret adjusted predictions and marginal effects",
            "journal": "Econometric Software Journal",
            "volume": "12",
            "issue": "2",
            "pages": "308-331",
            "doi": "10.1177/1536867X1201200209",
            "bibtex": r"""@article{williams2012using,
  author    = {Williams, Richard},
  title     = {Using the margins command to estimate and interpret adjusted predictions and marginal effects},
  journal   = {Econometric Software Journal},
  year      = {2012},
  volume    = {12},
  number    = {2},
  pages     = {308--331},
  doi       = {10.1177/1536867X1201200209}
}"""
        }
    ]
}

# Aliases
ALIASES = {
    "reg": "regress",
    "ols": "regress",
    "fe": "xtreg",
    "re": "xtreg",
    "2sls": "ivregress",
    "iv": "ivregress",
    "kaplan": "sts",
    "km": "sts",
    "cox": "stcox",
    "ph": "stcox",
    "hp": "tsfilter",
    "hpfilter": "tsfilter",
    "estat_brant": "brant",
    "ologit": "brant",
    "estat_weakiv": "weakiv",
    "survey": "svy",
    "did": "csdid"
}


def lookup_citations(key: str) -> Optional[List[Dict[str, str]]]:
    """Looks up citations for a given command name or model type."""
    norm = key.strip().lower()
    norm = ALIASES.get(norm, norm)
    return CITATIONS_DB.get(norm)


def format_citation_response(key: str, entries: List[Dict[str, str]]) -> str:
    """Formats human-readable citations and BibTeX block for display."""
    canonical = ALIASES.get(key.strip().lower(), key.strip().lower()).upper()
    title_str = f"{canonical} ({key.upper()})" if canonical != key.upper() else canonical
    lines = [
        f"Academic Citation & Reference: {title_str}",
        "=" * 72,
        ""
    ]
    for i, e in enumerate(entries, 1):
        lines.append(f"[{i}] {e['author']} ({e['year']}).")
        lines.append(f"    \"{e['title']}\".")
        lines.append(f"    {e['journal']}" + (f", {e['volume']}({e['issue']}): {e['pages']}." if e.get('volume') else "."))
        lines.append(f"    DOI: https://doi.org/{e['doi']}")
        lines.append("")
        lines.append("BibTeX Entry:")
        lines.append(e['bibtex'])
        lines.append("-" * 72)
        lines.append("")
    return "\n".join(lines).strip()
