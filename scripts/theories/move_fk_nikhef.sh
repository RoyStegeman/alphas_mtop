#!/bin/bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

#
#for j in 10 11 12; do
#  for i in {00..12}; do
#    THEORY_ID="400${j}0${i}"
#    echo "Downloading theory_${THEORY_ID}..."
#    # Download the theory_4000x000.tgz file
#    curl -u :$PASSWORD -o "theory_${THEORY_ID}.tgz" "https://cernbox.cern.ch/remote.php/dav/public-files/MsBioAS1FShShI9/theory_${THEORY_ID}.tgz?signature=f59683a0c2f9d67e4c7efca76398e4345981ca949002cb05f2fe69c5c1350f25&expiration=2025-05-08T15%3A39%3A56%2B02%3A00" -#
#
#    # move to nikhef
#    #scp -P 2022 "theory_${THEORY_ID}.tgz" jthoeve@erf.nikhef.nl:/var/www/html/nnpdf/theories
#
#
#  done
#done

THEORY_IDS=(40009000 40009001 40009002
40010000 40010001 40010002 40010003 40010004 40010005 40010006 40010007 40010008 40010009 40010010 40010011 40010012
40011000 40011001 40011002 40011003 40011004 40011005 40011006 40011007 40011008 40011009 40011010 40011011 40011012
40012000 40012001 40012002 40012003 40012004 40012005 40012006 40012007 40012008 40012009 40012010 40012011 40012012)

for THEORY_ID in "${THEORY_IDS[@]}"; do

    #echo $THEORY_ID

    #echo "Downloading theory_${THEORY_ID}..."
    # Download the theory_4000x000.tgz file
    #curl -u :$PASSWORD -o "theory_${THEORY_ID}.tgz" "https://cernbox.cern.ch/remote.php/dav/public-files/MsBioAS1FShShI9/theory_${THEORY_ID}.tgz?signature=f59683a0c2f9d67e4c7efca76398e4345981ca949002cb05f2fe69c5c1350f25&expiration=2025-05-08T15%3A39%3A56%2B02%3A00" -#

    # move to nikhef
    echo "Creating new theory_${THEORY_ID}.tgz file..."
    tar -czf "theory_${THEORY_ID}.tgz" "theory_${THEORY_ID}"
    scp -P 2022 "theory_${THEORY_ID}.tgz" jthoeve@erf.nikhef.nl:/var/www/html/nnpdf/theories


done