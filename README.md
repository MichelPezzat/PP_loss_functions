# PP_loss_functions

An open source implementation of the multiple time-step loss functions for peak-picking proposed in [1].

A cnnSA3F5 model [1,2] is utilised with CE, MI, MMD and WMD loss functions for automatic drum transcription using data from [3]. 

## Licenses

#### Code

This code is published under the BSD license which allows redistribution and modification as long as the copyright and disclaimers are contained. 

#### Data

The example data is taken from the [MDB Drums dataset](https://www.github.com/CarlSouthall/MDBDrums) [3] which is published under a  [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

The full license information can be found on the [license](https://github.com/CarlSouthall/PP_loss_functions/blob/master/LICENSE) page. 


## Required Packages

• [numpy](https://www.numpy.org)   
• [tensorflow(1.8.0)](https://www.tensorflow.org/)      
• [matplotlib](https://matplotlib.org/)

## References


| **[1]** |                  **[C. Southall, R. Stables and J. Hockman, Improving Peak-Picking Using Multiple Time-step Loss Functions, Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR), Paris, France, 2018.](https://carlsouthall.files.wordpress.com/2018/08/pp_loss_functions.pdf)**|
| :---- | :--- |

| **[2]** |                  **[C. Southall, R. Stables and J. Hockman, Player Vs Transcriber: A Game Approach To Data Manipulation For Automatic Drum Transcription, Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR), Paris, France, 2018.](https://carlsouthall.files.wordpress.com/2018/08/player_vs_transcriber.pdf)**|
| :---- | :--- |

| **[3]** |                  **[C. Southall, C. Wu, A. Lerch, J. Hockman, MDB Drums - An Annotated Subset of MedleyDB for Automatic Drum Transcription, Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR), Suzhou, China, 2017.](https://carlsouthall.files.wordpress.com/2017/12/ismir2017mdbdrums.pdf)**|
| :---- | :--- |


