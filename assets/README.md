# Figures

| File | Source | Used for |
| --- | --- | --- |
| `masc_overview.png` | Figure 2 of the paper | The overview diagram in the root [README](../README.md) |
| `masc_overview.pdf` | Figure 2, vector original | Source for the PNG; re-render if the figure changes |

GitHub cannot display a PDF through an `<img>` tag, so the README references the
PNG and the PDF is kept alongside it as the editable vector source.

To re-render after editing the PDF (macOS, no extra tooling required):

```bash
qlmanage -t -s 2400 -o /tmp assets/masc_overview.pdf
mv "/tmp/masc_overview.pdf.png" assets/masc_overview.png
```

The PDF's CropBox is already tight around the figure, so the render needs no
trimming. 2400 px wide gives a crisp result at the README's 900 px display width
on high-DPI screens.
