# Scoring functions

## Step-by-step guide

1. Download the following, rename, and place into `data/raw/lexical_norms`:
    <table><tr>
    <th> Link to download </th> <th> Rename file to </th>
    </tr><tr><td>

    * https://osf.io/d7x6q/files/vb9je
    * https://osf.io/ch48r/files/ma586?view_only=ca45900ffc264645a32394d256101e7d
    * https://link.springer.com/article/10.3758/s13428-013-0403-5#Sec10 (Under Supplemental Material)
    * https://osf.io/ksypa/files/he4dv

    </td><td>

    * KupermanAoA.xlsx
    * AIGeneratedAoA.xlsx
    * BrysbaertConc.xlsx
    * MultiwordConc.csv

    </td></tr></table>

2. Make sure all packages in pyproject.toml are installed (including lexical_features)
    ```cmd
    .../age-of-acquisition> pip install -e .
    ```

2. Usage
    ```python
    from lexical_features import *

    # age of acquisition
    print(aoa("dog"))

    # concreteness (supports multiword expressions)
    print(conc("dog"))
    print(conc("bite the bullet"))

    # word frequency
    print(freq("dog"))

    # phonological complexity
    print(phon("dog"))
    ```

4. Function details
    * aoa(word) ∈ [1.58, 20.6]
        * Curriculum follows increasing AoA
    * conc(word) ∈ [1.0, 5.0]
        * Curriculum follows decreasing Conc
    * freq(word) ∈ (0, 8]
        * Curriculum follows decreasing Freq
    * phon(word) ∈ []
        * Curriculum follows increasing Phon
    
    All scoring functions return -1 if word is not found
    ...

5. Additional notes
    * Inflectional variants already added to datasets
    * Ineligible words like stopwords are still scored; remove them in
      sentence level scoring if desired