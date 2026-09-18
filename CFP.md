# CFP tracker

Conference submissions for this project.

| Conference | Deadline | Submitted | Status |
|---|---|---|---|
| iOS Conf SG (Singapore, 20–22 Jan 2027) | 30 Sep 2026 | Not yet | Draft ready, not yet submitted |
| Arctic Conference | 16 Oct 2026 | Not yet | Not started |
| AppDevcon | 18 Dec 2026 | Not yet | Not started |

## Bio

> Ixhen Hasani is an iOS engineer in Munich. He architected and built the
> CarPlay layer of the Lamborghini Unica app in pure SwiftUI rather than
> CarPlay templates. In production since October 2025 on the Urus, and
> coming to the Temerario from the same codebase. Six years of shipping
> iOS software inside cars, where the hardware fights back.

To verify before submitting anywhere: whether the Temerario version should
be described as "coming to" vs. already shipping.

## iOS Conf SG

**Title:** Attention Is All You Need. 100 Seconds Is All You Get.

**Abstract:**

> You want to ship an AI model in your iPhone app. Can you? I couldn't
> find a straight answer anywhere, so I measured it: 25,000 inferences of
> Whisper's encoder on a physical iPhone under controlled conditions, with
> the raw data published.
>
> Three things I found. A 39MB Core ML model costs 8MB of RAM, not 39.
> Core ML memory-maps weights, so every capacity estimate based on file
> size is wrong by 4–6x. Latency holds flat for about 100 seconds and then
> steps off a cliff; the "gradual thermal degradation" everyone assumes is
> actually one sharp transition to a different machine. And 4-bit
> quantization makes the model 4x smaller and 3% faster while nearly
> tripling the word error rate.
>
> I also found four ways my own benchmark was lying to me. Timing load and
> inference together made me wrong by a factor of 48. Benchmarking on a
> charging phone read 20% slow and invented a warm-up curve that doesn't
> exist. And the summary statistic I trusted said "slope" when the raw
> data said "cliff."
>
> This talk is what actually runs on a phone, how to find out for
> yourself, and why most published Core ML numbers are measured wrong.
> Everything is open source, including the limitations section, which is
> longer than I'd like.

To verify before submitting: that 25,000 is right (12,848 + 12,561 from
the two sustained runs, plus the quick tests, if anything conservative).

## Arctic Conference

**Title:** _(TBD)_

**Abstract:** _(TBD)_

## AppDevcon

**Title:** _(TBD)_

**Abstract:** _(TBD)_
