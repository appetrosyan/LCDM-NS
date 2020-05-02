(TeX-add-style-hook
 "presentation"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("beamer" "bigger")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem")))
   (TeX-run-style-hooks
    "latex2e"
    "./illustrations/mixture_2"
    "./illustrations/mixture_3"
    "./illustrations/mixture_4"
    "./illustrations/benchmark"
    "./illustrations/evidence-drift"
    "beamer"
    "beamer10"
    "inputenc"
    "fontenc"
    "graphicx"
    "grffile"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "textcomp"
    "amssymb"
    "capt-of"
    "hyperref"
    "natbib"
    "bm"
    "pgfplots"
    "dsfont"
    "xcolor"
    "listings")
   (TeX-add-symbols
    "TopHat"
    "CDF"))
 :latex)

