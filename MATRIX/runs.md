| Dataset                                  | Grid | Pre | Main 172.5 | Main 170 | Main 175 |
|------------------------------------------|------|-----|------------|----------|----------|
| ATLAS_TTBAR_13TEV_HADR_DIF               | X    | X   |            |          |          |
| ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR | X    | X   |            | R        |          |
| CMS_TTBAR_5TEV_TOT_X-SEC                 | X    | X   |            |          |          |
| TTBAR_7TEV_TOT_X-SEC                     | X    | X   |            |          |          |
| TTBAR_8TEV_DIF                           | X    | X   |            | R        |          |
| TTBAR_13TEV_DIF                          | X    | X   |            | X        |          |


Remember to put include_pre_in_results = 0 for the mt 175 runs

ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR is only to get the double differential distribution: the y binning
is different at each mtt bin

v2

| Dataset                                  | Grid | Pre | Main 172.5 | Main 170 | Main 175 |
|------------------------------------------|------|-----|------------|----------|----------|
| ATLAS_TTBAR_13TEV_HADR_DIF               | R    |     |            |          |          |
| ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR | R    |     |            |          |          |
| CMS_TTBAR_5TEV_TOT_X-SEC                 | R    |     |            |          |          |
| TTBAR_7TEV_TOT_X-SEC                     | X    | X   | X          |          |          |
| TTBAR_8TEV_DIF                           | R    |     |            |          |          |
| TTBAR_13TEV_DIF                          | X    | X   |            | R        |          |

