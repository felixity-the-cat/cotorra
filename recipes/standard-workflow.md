## Full workflow

<details>

<summary>0. Localize filenames by cluster.</summary>

```sh
case "$(uname -n)" in
    cri*)
        hm="/gpfs/data/bbj-lab/users/burkh4rt"
        ;;
    bbj-lab*)
        hm="/mnt/bbj-lab/users/burkh4rt"
        ;;
    *)
        echo "Unknown host $(uname -n)"
        ;;
esac
```

</details>

1. We run [cocoa](https://github.com/bbj-lab/cocoa) on the full
   [MIMIC dataset](https://mimic.mit.edu) that's been converted to
   [CLIF](https://physionet.org/content/mimic-iv-ext-clif). We also use
   [clipy package](https://common-longitudinal-icu-data-format.github.io/clifpy/)
   to create the sofa table and process the respiratory support and medication
   tables.

   ```sh
   for s in mimic ucmc; do
      cocoa collate \
         --raw-data-home "${hm}/data-raw/${s}-2.1.0/" \
         --processed-data-home "./processed/${s}" \
         --verbose
   done

   cocoa tokenize \
       --processed-data-home ./processed/mimic \
       --verbose

   # apply mimic-native tokenizer to ucmc
   cocoa tokenize \
       --processed-data-home ./processed/ucmc \
       --tokenizer-home ./processed/mimic/tokenizer.yaml \
       --verbose

   for s in mimic ucmc; do
      cocoa winnow \
         --processed-data-home "./processed/${s}" \
         --verbose
   done
   ```

2. Next we tune a model on this data:

   ```sh
   cotorra tune \
       --processed-data-home ../cocoa/processed/mimic \
       --output-home ./output/mimic/ \
       --verbose
   ```
