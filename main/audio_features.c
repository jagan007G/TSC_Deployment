#include "audio_features.h"

#include <math.h>
#include <stdlib.h>

void extract_features(
    int16_t *audio,
    int samples,
    float *features
)
{
    float rms = 0;
    float peak = 0;
    float mean = 0;
    float zcr = 0;

    for (int i = 0; i < samples; i++)
    {
        float s = audio[i] / 32768.0f;

        rms += s * s;

        mean += fabsf(s);

        if (fabsf(s) > peak)
        {
            peak = fabsf(s);
        }

        if (i > 0)
        {
            if ((audio[i] > 0) != (audio[i - 1] > 0))
            {
                zcr++;
            }
        }
    }

    rms = sqrtf(rms / samples);

    mean /= samples;

    zcr /= samples;

    float crest = peak / (rms + 1e-6f);

    features[0] = rms;
    features[1] = zcr;
    features[2] = peak;
    features[3] = mean;
    features[4] = crest;

    // Placeholder features
    features[5] = 0;
    features[6] = 0;
    features[7] = 0;
    features[8] = 0;
}