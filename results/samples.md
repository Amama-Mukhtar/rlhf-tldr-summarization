# Qualitative samples (10 evaluation posts)

Post text is the excerpt saved by `evaluate.py` (end of the post). r = reward-model score.
Posts 195, 13, 277 and 264: PPO or DPO better than SFT. Posts 228, 210, 223 and 258: worse.

## Sample 1: post #195 (better)

```
he full service. Contact members, the whole deal. Idiot.

But wait, that's not all! With this deal that already includes stupidity, ignorance and an immature teenage mind, I'll also throw in a girlfriend of over a year!

Why I did this, I do not know. I love my girl. I've got this weird thing... I just call it a personality flaw. I have a fear of... Missing out on something I guess. I've had it with all sorts of things from sport to school to... this now. She's the only girl I've been with, that's the only thing I can think of that drove my stupid brain to do such a dumb thing. I want to erase everything that's happened the last 48 hours! God damn. Sorry for blurting at the end there.
TL;DR:
```

- **Human:** ? Please read some at least. I need some help here. I'm in a bad way, feel terrible. Sort of hate myself.
- **SFT** (r=-4.73, len=48): I'm a girl, I'm a girl, I'm a girl, I'm a girl, I'm a girl, I'm a girl, I'm a girl, I'm a girl, I'm a girl, I'm a
- **PPO** (r=-0.54, len=48): There's something wrong with me after paying a girlfriend of over a year for trial of full service after paying a girlfriend of over a year for trial of full service after paying a girlfriend of over a year after paying a girlfriend of over a year
- **DPO** (r=1.48, len=42): I joined a website that includes stupidity, ignorance and an immature teenage mind. I threw in a girlfriend of over a year. I want to erase everything that's happened the last 48 hours!

**Why:** SFT is stuck repeating "I'm a girl" (also wrong). DPO keeps the website, the girlfriend and the regret, though it copies odd phrases. PPO repeats itself.

## Sample 2: post #13 (better)

```
s I get to know them and they get to know me and we do   stuff.

I'm just tired of this high school bull shit, I just want someone to love. Someone to cuddle without having sex, without thinking that sex makes are relationship serious, I mean I just want a girl who doesn't think that all guys want is sex. I need help finding the kind of girl whose beauty lies within and not only on the outside, a girl i can give flowers to randomly and she'll realize how special she is, a girl I can lay on my roof with and look up at the stars and fall asleep on each other. I guess I really just wanna know how to find great girls who want those amazing relationships and I need help find myself a bit.t
TL;DR:
```

- **Human:** I really need help in being able to better my relationship and finding myself so I can truly give all of myself to my eventual significant other
- **SFT** (r=-2.93, len=48): I want a girl who doesn't think that all guys want is sex, a girl who doesn't think that all guys want to be with her, a girl who doesn't think that all guys want to be with her, a girl who
- **PPO** (r=0.81, len=48): I need help finding the kind of girl whose beauty lies within and not only on the outside, a girl i can give flowers to randomly and she'll realize how special she is, a girl i can lay on my roof with and look up
- **DPO** (r=2.73, len=48): I'm 17 and I feel like while I'm good looking I can't find any girl who cares enough to get to know me for me. I need help finding the kind of girl whose beauty lies within and not just on the

**Why:** SFT loops on one phrase. DPO and PPO state the actual request: help finding the right kind of girl. Both are cut off at 48 tokens.

## Sample 3: post #277 (better)

```
ged her. Things were going well, we hit it off and had even set up a date. A couple of days after she was at a party and stayed on the phone to me drunk for most of the night, so she was obviously into me.

Then the fuck up happened. 

It isn't dramatic, just stupid. I felt pretty bad for trying to date so soon, and I told her this, explained that I'd just broken up with my GF, and that I still had feelings for her etc, but that I thought she was a really lovely girl and that I'd really like to get to know you more. She didn't see it that way and saw herself as a rebound. So I lost out on getting to know a really nice person who I clicked really well with because I tried being honest.
TL;DR:
```

- **Human:** Broke up with GF of three years, tried dating, fucked up by telling new girl I still had feelings for ex.
- **SFT** (r=-1.68, len=48): I told a girl I really liked about my recent break up with my GF, she didn't see it that way and I lost out on getting to know a really nice person who I clicked really well with because I tried being honest.
- **PPO** (r=4.10, len=48): I broke up with GF of 3 years and found her on FB and messaged her about feelings for me after breakup, she saw it as a rebound and lost out on getting to know a nice person who I clicked really well
- **DPO** (r=2.29, len=40): I broke up with my ex, told her I really liked her, she didn't see it that way and lost out on getting to know a really nice person who I clicked really well with.

**Why:** DPO keeps the main story in fewer tokens and adds the break-up. It leaves out the three years and the ex-feelings detail. SFT is close, so this gain is small.

## Sample 4: post #264 (better)

```
seful?
POST: When I was a kid, I've always wanted to be a rebel with an extraordinary life who made an impact. Unfortunately, I ended up in a vanilla state of things: I'm F/21, living in US, working a full time job, and being part-time in school. No SO at this point. I am not satisfied with where I am. I've always wanted more and I feel like now is the time for more.

My job pays me well so I have reasonable financial resources. I create my own schedule at work so my time is fairly flexible. Give me ideas - anything I can do on either a daily, weekly, or monthly basis (but something that doesn't interfere too much with my daily life; e.g. I can't move to a different state or country).
TL;DR:
```

- **Human:** I have a great life, but it's kind of boring... I want to be more impactful and have a more fulfilling life. What do you challenge me to do for this following year?
- **SFT** (r=-3.85, len=17): I want to be a rebel, but I don't know what to do.
- **PPO** (r=0.39, len=48): What would you dare or challenge me to do for a year starting today, to make my life more fulfilling and more purposeful? My job pays me well so I have reasonable financial resources. What I do now is relatively flexible and allows me to
- **DPO** (r=2.60, len=48): I want to be a rebel with an extraordinary life who makes an impact on my life and my career. I want to do something that doesn't interfere too much with my daily life. Anything I can do on either a daily, weekly,

**Why:** SFT misses the request. PPO asks for a challenge for the year, which is the real question. DPO only gets part of it.

## Sample 5: post #228 (worse)

```
ow.

He has agreed, several times after doing something awful, to go to therapy. He has never followed through, however, insisting he "doesn't have time" or "doesn't need to."

Over the course of our relationship we've seen each others' highs and lows. For a while he was able to distance himself from his dysfunctional family and was a much happier person for it. Now his dad (who still drinks heavily but is too weak to be physically abusive) is dying (very slowly) and things have gotten quite bad again.

Reddit, what can I do to help my fiancé? I don't want to believe this is abusive lost cause, but I can't get married to someone I can't trust, and right now there is very little trust.
TL;DR:
```

- **Human:** My fiancé causes many unnecessary problems in our relationship by avoiding conflict, lying, and abusing alcohol. How can I help him?
- **SFT** (r=1.93, len=43): My fiancé is a great person, but he has a lot of conflict-avoiding tendencies, he lies compulsively, and he cannot control his drinking. How can I help him through this?
- **PPO** (r=-2.80, len=48): My fiancé has alcohol addiction issues and alcoholism issues and alcohol abuse issues and alcohol addiction issues and alcohol abuse issues and alcohol addiction issues and alcohol addiction issues and alcohol addiction issues and alcohol addiction issues and alcohol addiction issues and alcohol addiction issues
- **DPO** (r=-1.77, len=48): My fiancé is a great person, but he has a lot of conflict-avoiding tendencies, he lies compulsively, and he cannot control his drinking. How can I help him through this? I don't want to believe

**Why:** SFT covers the fiancé, lying and drinking and asks for help. PPO collapses into repeating "alcohol addiction issues". DPO is SFT plus an unfinished sentence.

## Sample 6: post #210 (worse)

```
ice for an unknown problem
POST: So me and a group of 7 friends were going to go see Wicked (a Broadway play) tonight. I was all set to go until I got a call from one of the friends in question. He told me that one of the girls in the group had some shit with me and wouldn't go if I was going to be there. Not wanting to cause any trouble I decided that I wouldn't go.

But what I want to know, is why she had shit with me. Me and her have never been the closest friends and I don't believe I've said anything that could have upset her. In fact, I haven't really hung out with her in a long while. Any ideas of why she might not want to see me, Reddit? Any advice on how to get closer to her?
TL;DR:
```

- **Human:** A girl has some unknown "shit" with me and doesn't want to see me when I haven't done anything that might have upset her.
- **SFT** (r=0.07, len=22): I'm going to see Wicked tonight and I want to know why she had shit with me.
- **PPO** (r=-4.66, len=48): What happened when we went to see Wicked (a Broadway play) tonight, what happened when we went to see Wicked (a Broadway play) tonight, what happened when we went to see Wicked (a Broadway play) tonight,
- **DPO** (r=-1.00, len=48): I was all set to go to see Wicked (a Broadway play) tonight and was all set to hang out with a girl who had shit with me. I was all set to go but she had shit with me. Any

**Why:** PPO repeats the same clause and never mentions the girl's problem. SFT is short but keeps both points.

## Sample 7: post #223 (worse)

```
." (While it doesn't, you just have to trust me on this one) just to avoid physical interaction. It's been getting to me recently, and it's actually bringing my confidence down about myself. I don't know if I'm ugly, or I'm clingy... it's just mind boggling. I've tried talking about this with her and she simply says, "I just don't like it". But honestly I feel as if that's an excuse and this has been ongoing for about 7-8 months. I get jealous of seeing something as simple as couples that simply hug and give a peck when they see each other, I get none of that except a cold "hello", it's just not the same anymore and I don't know what to do. I don't even know if she loves me anymore...
TL;DR:
```

- **Human:** girlfriend shows no affection for no reason. I want it, dunno if I'm ugly clingy or god knows what, have tried talking about it, no dice. halp
- **SFT** (r=1.31, len=25): My girlfriend of almost two years now shows no affection whatsoever, and it's depressing. I'm not kidding.
- **PPO** (r=-3.09, len=48): My girlfriend of almost two years now shows no affection whatsoever, and it's depressing me constantly, what do I do? What do I do? What do I do? What do I do? What do I do? What do I
- **DPO** (r=-1.71, len=48): My girlfriend of almost two years now shows no affection whatsoever, and it's depressing. I don't know if I'm ugly, or clingy... it's just mind boggling. I'm not kidding when I say kiss

**Why:** PPO starts well, then repeats "What do I do?" to the token limit. SFT is short and clear.

## Sample 8: post #258 (worse)

```
und, Ive had the flu for about 4 days. The first few days he texted me, everything was fine. Then yesterday, he didn't text me at all, all day. When I texted him, he admitted to just... forgetting about me.

Now, I don't demand much in the area of texting. Once a day before bed, a quick "I love you." We used to talk all the time, he was like my best friend. Now... I'm lucky to even get a text between all his video games and stuff. We see each other at school for maybe 10 minutes tops. He claims "there's nothing to talk about".

I don't know how to respond. I think about him constantly. He said hes sorry and that he loves me, I don't doubt that. I just don't know where to go from here.
TL;DR:
```

- **Human:** I'm sick, boyfriend of a year forgot about me for an entire day. Communication has dwindled as it is. Don't know where to go from here.
- **SFT** (r=0.92, len=24): My boyfriend of 1 year forgot about me for an entire day. I don't know how to respond.
- **PPO** (r=-3.17, len=48): My boyfriend forgot about me for an entire day yesterday texting me for 10 minutes after that he texted me again, what do I do? What do I do next? Should I contact him? What do I do next?
- **DPO** (r=1.20, len=46): My boyfriend forgot about me for an entire day, texted me all day, I don't know how to respond. He said he loves me, I don't doubt that. I don't know how to respond.

**Why:** SFT has the fact and the question. PPO invents a ten-minute texting story and repeats itself. DPO contradicts itself about texting.

## Sample 9: post #273 (judge fooled)

```
ss is extremely flexible because of my internship situation and my permanent job hunt. It's also very close to where I live.

Now, I've been offered a job at a different shop a little bit further from where I live. However this isn't available until October. But I would expect I'd at least be paid minimum wage, however I don't know how flexible the manager will be (although I've got a good rapport with them), and the opening hours are earlier and later than my current job and I'd prefer not to do those sorts of hours.

By the time this problem will arise, hopefully I will have found a permanent job and won't need a part-time job. However I'd like to get a bit of perspective on it now.
TL;DR:
```

- **Human:** will be offered a better part-time job which won't have any of the negatives of my current job but might not have the benefits.
- **SFT** (r=-0.45, len=37): I'm doing a part-time job at a shop I don't know much about, I'm not sure if I should be offered a job at a different shop or not.
- **PPO** (r=5.08, len=48): I'm doing an internship in the shop I want to go into seasonally pay below minimum wage and need a temporary job opening hours earlier than my current job to ensure I can get a bit of perspective on the potential benefits of a temporary job opening
- **DPO** (r=1.97, len=48): I'm doing an internship in the field I want to go into, it's unpaid so I have a part-time retail job to help support myself, I don't have many hours and the manager is extremely flexible. What are my options

**Why:** r=5.08 is the highest PPO score shown, but the text is a run-on mixture of two jobs. This is a reward-model mistake.

## Sample 10: post #141 (mixed)

```
s only seen the word in books but has never heard it spoken" would say it. Then he mocked me by saying the word the way I pronounced it a few times. 

He also likes to correct others as well. We play in a couple of role-playing games, and he had to point out that someone made a mistake by saying "consecrated" instead of "desecrated", and laughing about it. I told him that the way he was correcting people was rude, but he still does it all the time. 

My question is, what can I do to get him to stop? It makes him look like a jackass, but he's honestly a great person and I don't want people to get the wrong impression of him. Apparently, outright telling him that it's rude did not work.
TL;DR:
```

- **Human:** Super smart boyfriend corrects everyone, myself included. I told him that the habit is rude, but it hasn't stopped him. What can I do to make him quit?
- **SFT** (r=1.46, len=37): Boyfriend is a great person, but when I make a grammar/spelling/math mistake, he doesn't correct me. What can I do to get him to stop?
- **PPO** (r=-2.77, len=48): How can I get my boyfriend (31/M) to stop correcting me (27/F) and others? Flat-out asking didn't work. What can I do to get him to stop?? What can I
- **DPO** (r=1.73, len=43): My boyfriend is smart, but when I make a grammar/spelling/math mistake, he doesn't correct me immediately, but laughs at me instead. I'm trying to get him to stop.

**Why:** SFT and DPO change the meaning (they say he does not correct). PPO has the right question, repeated, yet the judge gave PPO the lowest score.
