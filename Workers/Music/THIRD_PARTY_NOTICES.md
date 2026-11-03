# Third-party notices

## mere.run

Parts of ACEStepMusic's network code (the attention, encoder, quantizer, detokenizer and DiT layer structure, the configuration fields and the constrained-decoding state names) were adapted from mere.run's Swift ACE-Step implementation (github.com/sawfwair/mere-run, MereRunCore/ACEStep), under this licence:

```
MIT License

Copyright (c) 2026 Sawfwair Inc. and mere.run contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ACE-Step 1.5

The port follows github.com/ace-step/ACE-Step-1.5 at ca1e85f (MIT); the weights are ACE-Step/Ace-Step1.5 on Hugging Face (MIT), fetched from the byte-identical copy at arraypress/song-ace-step-1.5:

```
MIT License

Copyright (c) 2026 ACEStep

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## YuE2 inference code (m-a-p, Apache-2.0)

YuE2Music is a Swift/MLX implementation that follows m-a-p's YuE2 inference code, yue2-infer
(https://github.com/multimodal-art-projection/YuE, 0.1.5 as shipped in m-a-p/YuE2-3B at
8f24312; 0.1.6 at commit 1647252d68b70cbead046ffd3f7caf036dde47ee declares its licence):
its prompt protocol (token ids and instruction lines), its sampling rules and its NAR
midpoint flow-matching solver. Copyright the YuE authors, Multimodal Art Projection (m-a-p).
Licensed under the Apache License, Version 2.0; you may obtain a copy at
http://www.apache.org/licenses/LICENSE-2.0 (the full text is in that repository's
LICENSE). No Python source is included here; the Swift is new and is distributed under this
package's MIT licence, with this notice for the work it follows.

## YuE2 weights (m-a-p, CC BY-NC 4.0 with an individual-creator permission)

YuE2Music runs the YuE2-3B, YuE2-Vae and YuE2-Vae-legacy weights
(https://huggingface.co/m-a-p/YuE2-3B). They are not part of this package: they are fetched
by a person who has accepted their terms. They are licensed under Creative Commons
Attribution-NonCommercial 4.0 International (https://creativecommons.org/licenses/by-nc/4.0/legalcode)
together with the licensors' additional permission of 2026-09-16 (MODEL_LICENSE at the
commit above): individuals may use the weights free of charge and monetise their outputs;
companies need a commercial licence from the licensors; responsible-use conditions apply;
the weights themselves may not be sold or redistributed commercially. `YuE2Licence` carries
the permission verbatim and `song models licence yue2` prints it.

The YuE2 decoder is the Oobleck VAE with SnakeBeta activations. In this package that network
(MusicNetworks) is shared with ACE-Step, and upstream's notice for it reads: "The Oobleck VAE
and SnakeBeta implementation in modeling_vae.py is derived from stable-audio-tools commit
a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45."

```
MIT License

Copyright (c) 2023 Stability AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
MIT License

Copyright (c) 2022 NVIDIA CORPORATION.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
