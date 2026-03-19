"""Moral Geometry Benchmark v8b — Budget Edition ($50 quota)
Social Cognition Track | Measuring AGI Competition

Tests 5 geometric properties of moral cognition (Bond, 2026):
  T1. Structural Fuzzing — AITA data (needs verdict labels)
  T2. Bond Invariance Principle — Dear Abby data (no labels needed)
  T3. Holonomy — Dear Abby data (no labels needed)
  T4. Contraction Order — Dear Abby data (no labels needed)
  T5. Conservation of Harm — Dear Abby data (no labels needed)

Budget-optimized: 4 Gemini models (~$0.014/call) = ~$29 total.
Non-Gemini models cost ~$0.96/call and blow the $50 quota.
~500 calls/model × 4 models = ~2,000 calls = ~$28.

Paste this ENTIRE file into ONE cell in a Kaggle Benchmark Task notebook.
Expected runtime: ~60-90 min (adaptive, 4 Gemini models).
"""

import kaggle_benchmarks as kbench
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, json, time, random, math, threading

os.environ["RENDER_SUBRUNS"] = "False"

WORKERS_INIT = 50   # start aggressive
WORKERS_MIN = 2
WORKERS_MAX = 80

# Shared results store (kbench tasks must return None, not dict)
_results_store = {}


class AdaptivePool:
    """CSMA/CA-style adaptive concurrency.
    Starts at WORKERS_INIT, backs off on failure, ramps on success.
    """
    def __init__(self, initial=WORKERS_INIT, lo=WORKERS_MIN, hi=WORKERS_MAX):
        self._lock = threading.Lock()
        self.workers = initial
        self.lo = lo
        self.hi = hi
        self.successes = 0
        self.failures = 0
        self._window = 0  # calls since last adjustment
        self._adjust_every = 10

    @property
    def n(self):
        return self.workers

    def record_success(self):
        with self._lock:
            self.successes += 1
            self._window += 1
            if self._window >= self._adjust_every:
                self._adjust()

    def record_failure(self):
        with self._lock:
            self.failures += 1
            self._window += 1
            # Immediate backoff on failure (multiplicative decrease)
            self.workers = max(self.lo, self.workers // 2)
            print(f"    [ADAPT] failure → workers={self.workers}")
            self._window = 0
            self.successes = 0
            self.failures = 0

    def _adjust(self):
        fail_rate = self.failures / max(self._window, 1)
        if fail_rate == 0:
            # All success → additive increase
            self.workers = min(self.hi, self.workers + 5)
        elif fail_rate < 0.1:
            # Mostly success → small increase
            self.workers = min(self.hi, self.workers + 2)
        elif fail_rate < 0.3:
            # Some failures → hold steady
            pass
        else:
            # Many failures → multiplicative decrease
            self.workers = max(self.lo, self.workers // 2)
        self._window = 0
        self.successes = 0
        self.failures = 0

    def status(self):
        return f"workers={self.workers} ok={self.successes} fail={self.failures}"


_pool = AdaptivePool()

print("=" * 70)
print("MORAL GEOMETRY BENCHMARK v7d (adaptive parallelism)")
print("Five Geometric Tests of Social Cognition (Hybrid)")
print("Based on Bond (2026), Geometric Ethics")
print(f"Initial parallelism: {WORKERS_INIT} (adaptive {WORKERS_MIN}-{WORKERS_MAX})")
print("=" * 70)
print()

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD DATASETS
# ═══════════════════════════════════════════════════════════════════════

t0 = time.time()

# --- 1a. AITA dataset (for T1 — needs verdict labels) ---
print("[1a/8] Loading AITA dataset...")
try:
    from datasets import load_dataset
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

ds = load_dataset("OsamaBsher/AITA-Reddit-Dataset", split="train")
print(f"  Loaded {len(ds):,} AITA posts in {time.time()-t0:.0f}s")

random.seed(42)
PER_CLASS = 10  # 10 per class = 40 AITA scenarios for T1
pools = {"nta": [], "yta": [], "esh": [], "nah": []}

for row in ds:
    v = (row.get("verdict") or "").lower().strip()
    if v not in pools or len(pools[v]) >= PER_CLASS * 4:
        continue
    text = row.get("text") or ""
    score = row.get("score") or 0
    min_score = 5 if v in ("esh", "nah") else 10
    if len(text) < 300 or len(text) > 1500 or score < min_score:
        continue
    pools[v].append({
        "title": row["title"],
        "text": text[:1200],
        "verdict": v.upper(),
        "score": score,
    })

AITA_SCENARIOS = []
for v in ["NTA", "YTA", "ESH", "NAH"]:
    pool = pools[v.lower()]
    random.shuffle(pool)
    AITA_SCENARIOS.extend(pool[:PER_CLASS])
    print(f"  {v}: {len(pool[:PER_CLASS])}")

random.shuffle(AITA_SCENARIOS)
print(f"  AITA total: {len(AITA_SCENARIOS)} scenarios (for T1)\n")

# --- 1b. Dear Abby dataset (for T2-T5 — no labels needed) ---
# 50 curated morally complex letters (1985-2017), 2+ moral keywords each.
# Source: Kelly Garrett / The Pudding, "30 Years of American Anxieties"
print("[1b/8] Loading Dear Abby scenarios (embedded)...")

DEAR_ABBY = json.loads(r'''[{"title": "Widow Realizes the Perfect Gift Is Giving Love to Others", "text": "last week, my family suffered the loss of my grandfather. he was catholic, the only catholic in our immediate family, and his funeral was held in a catholic church as he wished. when it came time to receive communion, a family friend encouraged my grandmother and the rest of the non-catholic family members to receive communion. should we have received communion out of respect for our grandfather, or was it right to stand by our own beliefs? -- grieving in virginia", "year": 2005}, {"title": "A SECOND LOOK AT ABUSIVE FIANCE", "text": "i just finished reading the letter from the 19-year-old girl who signed herself \"second thoughts in moline, ill.\" abby, i could have written that letter myself. i'm also 19, have gone with a guy for nearly three years and we plan to be married soon, but i, too, have second thoughts for the same reasons. he's jealous, doesn't trust me, accuses me of going out with other guys, and we fight a lot about stupid things. i like people and enjoy having friends, but he always wants to be with me alone. lately he has cursed me, accused me of going out on him and has hit me. then he cries, begs me to give him another chance and swears he'll never do it again, but he does. my friends tell me i'll be sorry if i marry him, but, abby, i truly love him. can you tell me why women go right on loving guys wh", "year": 1988}, {"title": "Teen Directs Increasing Anger Toward His Mother and Sister", "text": "i'm afraid something is wrong with my younger brother. he's just turned 13, and he's become violent and angry. before, he used to tease me and our sister once in a while, but now it's becoming an everyday thing. if we tease him back, he gets mad and starts yelling. he directs most of his anger at our younger sister and our mom, and he has started to push my mom. she's worried that he might hurt one of us. if she confronts him when he gets out of hand, he starts yelling, \"what? i didn't do anything!\" then later, he'll come back and try to push or kick her. i have noticed that he talks to himself, too. we don't know what to do. we hear how kids sometimes harm their families. i'm afraid for my mom and my sister, but also for my brother. i love him and want to help him. -- afraid in arizona", "year": 2016}, {"title": "Therapist Frustrated by Parents Who Make Excuses for Kids", "text": "i have been dating \"charlie\" for a year. we are going to move in together at the end of the month. charlie is thoughtful and sweet, and for the first time in my life, i feel i can be totally myself around a man. last night, i had my feet propped up on his book bag and the bag fell open. i looked down and saw a pair of black women's panties in the style that he has been badgering me to wear. next to them were two dvds with pornographic pictures on the front. i was horrified. i have trusted charlie because he really doesn't have time to cheat on me. but why would he carry around another girl's underwear? abby, i can't think straight right now. i don't want to make a mistake, and i have no idea what to do. please help. -- scared to move in new york", "year": 2003}, {"title": "Boyfriend's Dating Profile Artfully Dodges the Truth", "text": "my boyfriend has posted his profile on a dating web site in the hope of finding some new friends. i am frequently out of town on business, and he has decided that he would like to converse with \"artsy\" people during the week while i am away. he claims this web site is the only way to meet like-minded people. while i don't mind his wanting to meet people, i feel that using a dating web site is inappropriate. i read his profile; in it he indicates that he is \"single.\" (he promises he will tell the woman he meets that he is not single \"when and if the topic comes up.\") i think it's wrong to meet people based on a lie. he swears he would never cheat on me. how can i convince him that this is a form of cheating and that it's disrespectful to me? -- frustrated in new york", "year": 2004}, {"title": "Moving Child's Grave Sparks Buried Anger After 20 Years", "text": "twenty-three years ago my husband and i lost our firstborn son. as my husband was active duty military, we could have buried him anywhere in the united states. at the time, we were in a place where my sister swore to me she would always live, and she would always be there to take care of him. i knew with my husband's career we had many more moves ahead of us, and it helped to ease the loss knowing that he would be taken care of. well, that lasted all of three years. my husband and i are now at a point where we have settled down and we know where we should have buried our precious angel, instead of trusting my sister. we want to have him exhumed, cremated and placed in a veterans cemetery, but my question is this: do i have the right to ask my sister to pay part of the costs as she \"broke\" ", "year": 2014}, {"title": "Grandma Feels Disrespected When Her Advice Is Ignored", "text": "whatever happened to respecting one's elders and recognizing grandparents as head of the family? i recently returned from visiting my son, his wife and my new grandchild. my son's wife and i had many disagreements on how to care for my grandchild. instead of respecting my years of experience as a mother and appreciating my help, she chose to ignore my instructions and advice. after i returned home, i was told by my son that i was no longer welcome to visit my grandchild unless i apologized to his wife for trying to undermine her parenting. i told him she should apologize to me for not showing me respect as the grandmother. how can i make my son see that it is his wife who is wrong, and not me? -- unappreciated grandma", "year": 2015}, {"title": "Mistress' Affair Has Ended After Death of Man's Wife", "text": "i am 52 years old and have been married for 22 years to my second husband. we have four teenage sons. i was widowed at 22 when my first husband was killed in vietnam. i was pregnant and lost our child when i was told of my husband's death. i was 30 when i married my second husband. he knows, of course, that i was married before, but for some reason i never felt confortable telling his parents. (my children know.) i think his parents always suspected something, but they've never asked me directly. my problem is, i am afraid this information will slip someday, and my in-laws will be hurt and angry at me. what do you think i should do? -- want to do the right thing", "year": 2002}, {"title": "Ex-Husband Who Drives Drunk Should Be Taken Off the Road", "text": "a few months ago i left my husband after a long marriage, mostly due to his drinking. he often insisted on getting behind the wheel while drunk, and i was uncomfortable about it, although i repeatedly begged him not to do it. since our split he has been drinking much later at his favorite bar. where he used to come home about 8, he now stays until 10 or 11. he recently had an accident on his way home from the bar, but managed to get away before the police arrived. part of me wants to contact the police and report it because i would feel horrible if he hurt someone and i had done nothing to stop it. i admit there are also selfish reasons i'd like to see him picked up. my concern is that he'll find out i turned him in. any suggestion on what to do? -- nervous in north carolina", "year": 2017}, {"title": "Surgeon General Calls Public to Combat Underage Drinking", "text": "a couple of weeks ago, some friends and i visited a family friend's niece who had recently had a baby girl. while we were visiting, we noticed that the baby was hungry. being a good mom, the new mother unbuttoned her shirt, took off her bra, and breast-fed the baby right in front of us. abby, was it right or wrong of her to expose her breasts in front of visitors when breast-feeding the child? -- rachel in philadelphia", "year": 2007}, {"title": "Wife Is Uncomfortable With Her Bisexual Fantasies", "text": "i'm a woman, twice married. my first marriage was to a woman who hurt me deeply by lying and cheating. i am now married to a man who, even with his faults, is a wonderful husband. my thing is, i am still strongly attracted to women. i consider myself to be bisexual. when my husband notices that i look at women, i'm honest and tell him what i admire about a particular woman. what i leave out is that i'm turned on by them. he is not open to my actively being bisexual, not even a threesome. is it all right for me to fantasize when i'm intimate with him that he's a woman? i know some people fantasize about being with a celebrity or a more attractive mate, but is it all right to fantasize about someone of a different gender? -- fantasizing in new york", "year": 2016}, {"title": "Woman Who Loves Two Losers Can't Decide Whom to Choose", "text": "i am so confused. i can't decide with whom i should spend the rest of my life. my ex-fiance, \"ramon,\" is in jail. ramon was a drug addict and is responsible for my bankruptcy. he swears he will be a changed man when he is released. there's also my ex-husband, \"fred.\" we were married for 10 years. he's the father of my two daughters. fred swears on a stack of bibles that he, too, has changed. both of them want me back. ramon is still very demanding, jealous and accuses me of cheating. believe me, i've had plenty of opportunities, but i haven't acted on any of them. fred has remarried, but says he will dump his wife to marry me. fred hit me a couple of times while we were together -- but truth be told, he is more of a mouse than a man. what should i do? i can't go to my family. they hate ram", "year": 2003}, {"title": "Theatergoer Has Reservations About Saving Latecomer's Seat", "text": "what do you think of the practice of \"reserving\" a seat at a public event by placing an object such as an umbrella or a coat on the seat? my feeling is this should not entitle a person to select a choice seat, then wander off for half an hour or more and expect others to respect the \"reservation.\" abby, will you please state in your column that saving a seat for someone who is late is very unfair and should not be permitted? also, how should a situation of this kind be handled? maybe you haven't been in a situation of this kind, but i'd like to hear from people who have. is it fair, or isn't it? and if the person who is \"holding\" a seat for a latecomer encounters an angry theatergoer, who is entitled to the seat? i have witnessed some ugly scenes as a result of \"seat saving\" in theaters. w", "year": 1997}, {"title": "Teens Racing to Be Parents Should Shift to Slower Gear", "text": "i am 16 years old and have a 5-month-old daughter. i thought her father and i would be together forever, but i was wrong. i was in love with him for more than two years. my problem is, i can't seem to find a boyfriend who is right for me. some boys don't mind that i have a baby, but all they want to do is go out with their friends. after a long day of feeding, changing and taking care of my daughter, i want someone at home to comfort me. is there anything wrong with that? -- lovesick in new york", "year": 2003}, {"title": "Tyke Becomes a Terror When Mom Takes Back Her Cellphone", "text": "when my friend \"fran\" and i get together with our kids, they often play games on her cellphone until the battery dies. if she tries to take the phone from her 6-year-old to make a call or recharge the phone, he starts yelling at her, pushes her, pulls her skirt and hits her. her reaction is to hug him and start praying for the devil to get out of his body in jesus' name as he continues to hit her. while i respect fran's religion, i'm appalled at his violent behavior, concerned that he will grow up thinking it's ok to hit people, and i think this should be handled differently. what do you think? should i say something? and if so, what can i say so as not to hurt her feelings? -- appalled by the violence", "year": 2014}, {"title": "Neighborhood Flasher Gives Woman Good Cause to Pause", "text": "my sister, \"emily,\" became engaged last week. she is planning her wedding, which will take place next year. emily's choice of a wedding date is causing a lot of hurt feelings among our family. she wants to be married on what would have been our father's birthday. daddy passed away while we were young, and it has been hard on the family. a lot of us feel she's being selfish to choose a day that belongs to our father and make it her own. emily insists that she's trying to honor daddy -- although some of her other actions suggest that she's acting out of spite for the rest of us. a lot of the family are saying they don't want to attend. i would hate to see my sister heartbroken on her wedding day, but do you find her choice of date appropriate or selfish? -- askance in southern calif.", "year": 2005}, {"title": "Runaway Sister's Poor Health May Put Her Life in Jeopardy", "text": "my father, who is in bad health, recently announced that he would like to be cremated and buried at the foot of my mother's grave. my birth mother died 28 years ago when i was 2, after they had been married only three years. dad married my stepmother when i was 8. i feel he should be buried with the wife he's been with for 22 years. she is the one who has seen him through the worst times in his life, his heart attack and stroke. my stepmother seems to have no negative feelings about it. am i wrong for thinking that a husband and wife should lie side-by-side when their time comes -- with a single headstone with their names and dates of birth/death/marriage? or is there some tradition i don't know about that he should be buried with his first wife? -- enquiring in clarkston, wash.", "year": 2009}, {"title": "GRANDMA MAKES THANK-YOU NOTES EASY", "text": "i am a 19-year-old girl who is very much in love with a guy i'll call billy. he is 22. i really thought we had a future together, but i never felt i could trust him completely. billy is very good-looking and can get any girl he wants. i wanted to test his faithfulness, so i asked tina-my best friend-to call up billy just to see if she could get him to go out with her. well, she did, and billy jumped at the chance. she said he didn't take her to any place special; they just rode around, got some burgers, then parked and made out. (just hugging and kissing.) i finally told billy that i had set the whole thing up with tina, and he got really mad at me. now he's going with tina, and i'm afraid i've lost him for good. abby, was i wrong to have done what i did? i really had to know. please don't", "year": 1994}, {"title": "HUSBAND REFUSES TO LET PETS IN BED", "text": "peter and i have been married for less than one year, and i am now faced with a problem that is threatening to break up our marriage. we are not kids. i am 45 and peter is 47. he absolutely will not allow any of our pets in bed with us. (we have a dog and two cats.) peter is extremely fastidious and says it's a matter of \"cleanliness.\" abby, our pets are well-groomed and they are just as clean as people. i had these pets before i married him, and they were always permitted on my bed, so now they are confused and hurt when they are not allowed on my bed. is there a solution? am i wrong to argue this point? i love my husband, but i think he's being unreasonable. please help me. my pets are so angry, they won't even look at me. animal lover", "year": 1988}, {"title": "Teacher's Idea of a Joke Is Student's Idea of a Dud", "text": "i need your opinion about something that happened at school. i am 13 years old, and my science teacher has an expression that bothers me. he says, \"life's unfair -- and then you die.\" he uses this expression whenever a student complains about something. he thinks it's funny. i know kids complain a lot, but i think he is wrong to say this. he makes it seem like life is hopeless. it makes me think about the boys in colorado who shot up their school, and about teen-agers who commit suicide. i think they felt hopeless, too. i would complain to the principal, but he knows about this, and he also thinks it's funny. what do you think? -- wondering in murrieta, calif.", "year": 2000}, {"title": "BRAGGING ABOUT PRICES CAN BE A COSTLY MISTAKE", "text": "i heard on the news that a 12-year-old boy was kicked out of the boy scouts because he didn't believe in god. i really got upset because i am a 12-year-old boy and i don't believe in god either. my friends don't respect me when they find out i don't believe in god. then they try to convince me that i am wrong. why can't they accept me the way i am? i don't go around telling people not to believe in god just because i don't. i don't think the boy scouts have the right to kick people out for their beliefs, do you? ticked in iowa", "year": 1985}, {"title": "Phone Call Won't Ease Guilt Caused by 20 Year Old Affair", "text": "i was pleased that you advised \"remorseful in georgia\" (jan. 27) to find another outlet for her guilt and \"leave the scab alone.\" i was recently contacted by my fiance's former girlfriend, a woman who had made several attempts to break us up when we first became a couple. although she apologized for the problems she tried so hard to cause between us, all it did was dredge the feelings of anger and anxiety up again. she was calling for purely selfish reasons -- not to give me the chance to confront her, but under the guise of \"wanting to be friends.\" whatever made her think i would want her friendship?! if \"remorseful\" needs a way to rid herself of her guilt, i recommend she get therapy. she may be trying to escape her karma. in my experience, she can run, but she can't hide. -- untouchable", "year": 2009}, {"title": "Bling on Bride's Finger Causes Husband Unease", "text": "please help me handle a problem with my brother-in-law, \"george.\" george has a dog that is aggressive toward people. \"brutus\" has bitten my nephews, nieces and several complete strangers. george brings brutus everywhere. he even brought brutus to our wedding, which was a formal event. i do not like brutus, and i'm afraid of what he might do to our 1-year-old child, the neighbors or to me. my husband and in-laws won't talk to george about this. am i wrong to expect my husband to step up and speak to his brother about brutus? i want to say something, but my husband always makes me feel like i'm being \"mean\" and that saying anything would hurt george's feelings. please help. -- dog-tired in missoula, mont.", "year": 2007}, {"title": "Hostess With the Mostest Has Guest Who's the Worst", "text": "i need to know if my husband's relationship with his ex-wife should be tolerated. they talk to each other on the phone every month or so, and send each other cards on special occasions. their closeness caused a former girlfriend to break off their relationship before we met. he is determined to stay close and sees nothing wrong with it. there were no children from the marriage, abby, so that is not the reason. why do people who remain this close get divorced? am i wrong to feel hurt and threatened, because i'm ready to just walk away from this warped, co-dependent relationship. please let me know your thoughts. -- ready to quit in arizona", "year": 2006}, {"title": "Brothers' Checkered History Remains Hidden From Family", "text": "i have been with my husband for 17 years -- married to him for 10 -- and we still have our ups and downs. two years ago i was drinking a lot. we separated for a few months, but still slept with each other occasionally. my husband ended up sleeping with a co-worker and got her pregnant. i was devastated; however, we worked it out and stayed together. but it's no longer the same. he tells me he loves me all the time, but sometimes i feel he's not happy with me and wants to be free. it's hard for me to trust him because he's still working with her. my husband tells me he doesn't see her that often because he works in a warehouse and she's in the office. but it still makes me feel insecure. how can i make things the way they used to be, before all of this? -- hurt in sacramento, calif.", "year": 2008}, {"title": "Brother in Law's Attachment to Kids Makes Mom Uneasy", "text": "i have known a certain 14-year-old girl, \"haley,\" since she was 7. i help take care of her now and then because her mother is a drug addict and is rarely around. haley lives at her friend's house, and she is starting to become sexually active. she goes very far, but hasn't gone all the way yet. would it be wrong of me to take haley to a birth control clinic and have the counselors speak with her and get her on birth control? the woman she lives with doesn't seem to care what the girl does and figures she shouldn't have to because it isn't her kid. this young lady needs to be steered in the right direction and i want to help. -- worried in bridgeview, ill.", "year": 2005}, {"title": "Couple Worries That Absence Won't Make Hearts Grow Fonder", "text": "i am an older bachelor who recently moved into a new home. i invited my neighbors -- a young married couple -- over for a home-cooked meal. they brought with them a lovely bottle of wine. i plan my dinners down to the last detail -- including selecting just the right wine to go with the meal. to make a long story short, i did not serve the wine my guests brought for our dinner. after thanking me for a wonderful meal and a delightful evening, they took the bottle of wine they had given me and went home! i didn't say anything, but am i wrong to be appalled by their rude behavior? -- mr. nice guy in tulsa", "year": 2002}, {"title": "Designer's High Success Can't Match Family's Expectations", "text": "if you have been asked this question before, please forgive me. i was wondering what the proper etiquette is about going out (not dating -- just appearing in public) after your husband dies. is there a waiting period? my husband passed away two weeks ago. i attended our church festival with two girlfriends, and i felt like i was being stared at. we didn't stay long. i am only 51 and my husband was 52. i know he would not have wanted me to stay at home -- but i want to do the right thing. -- newly widowed, baden, pa.", "year": 2000}, {"title": "Buying A House With Emergency Savings Threatens Man's Sense Of Security", "text": "while i was growing up, my parents taught me and my siblings to always keep a year's salary (pre-taxes) in a savings account that one never touches. the problem is my bride and i feel that we're ready to buy a home, although we don't have enough in our joint savings to make a down payment. she feels i should use my savings to make the down payment. i don't feel right about it because this savings technique has saved me twice in my life. once when i was a child and my parents lost their jobs, and again when i lost my job in the recession. am i selfish for wanting to keep my savings off limits? -- mr. savings", "year": 2014}, {"title": "Reader Has No Desire To Rekindle Friendship", "text": "an ex-friend of mine recently apologized for some bad behavior toward me, saying she had been going through a rough time. she wants to renew our friendship and said she misses it. i was taken aback and didn't know what to say. i replied, \"i'll get back to you about this,\" because i didn't want to hurt her feelings. abby, i have no desire to renew a friendship with her because i have had it with her volatile personality and her needy and clingy nature. how do i eventually respond? i was thinking of saying i have a full plate of responsibilities and commitments right now and can't make plans. i value your opinion, so what do you think? -- needs the right words in michigan", "year": 2013}, {"title": "Girl Wonders if Boyfriend's Shaking Could Lead to Abuse", "text": "i am a college student in a small town. eight months ago, i met a wonderful young man, and we were planning to be married until i told him about my past. my stepfather molested me. it was long ago, and i have since forgiven him and my mother. (mother is still married to him.) my boyfriend, however, cannot forgive them. he tried to convince my mother to leave my stepfather. she refused, and now my boyfriend and my mother no longer speak. he says things will never work out because of this rift he has with my family. i am willing to do whatever it takes to make the relationship work, but he says he can't be around my family, and it isn't fair to ask me to give them up. was i wrong to expect him to support my decision to forgive them? -- desperate in texas", "year": 2004}, {"title": "CHILD'S CRYING IS MUSIC TO HIS EARS", "text": "upon reading your column about a mother who gave away a gift her daughter had given her, let me tell you how i feel about it: many times i have given costly gifts to family--sons, daughters and parents. i've often bought them things that i would love to have had myself, but felt i couldn't afford. i would be much less hurt if they would tell me honestly that they had no use for my gift and would i mind if they gave it to so-and-so, or would i like to have it back? i once gave my daughter a very nice gift, and the next time i saw it, it was at her sister-in-law's. i was very hurt as i would rather have had it myself. would it be wrong when giving a gift to say, \"if you don't want this, will you please return it to me?\" hurt in florida", "year": 1987}, {"title": "Parents Object to Being Shut Out by Surgery Bound Daughter", "text": "my daughter, \"giselle,\" is scheduled to have serious surgery soon, and she has forbidden us to come to the hospital. she wants only her husband to be there. she has gone so far as to call us and make me promise that we will not come. she says we need to respect that she is a grown woman in her late 40s, and this is her decision and her way of dealing with the situation. giselle lives two hours from us, and she said she will let us know when we can visit for a few days. her husband will contact us as soon as the doctor talks to him after surgery. but giselle says that she simply \"does not want to be surrounded by family.\" i feel like we are being treated like family pets -- come when you're called; otherwise, stay out of the way. up to this point we had a close relationship with her. we can", "year": 2009}, {"title": "Family Feuds Over Passing of Plate From Bargain Buffet", "text": "i have a rare autoimmune disease that will end my life within a couple of years. after not dating for 15 years, i met a wonderful man. even though i tried not to, we fell in love. i think i should break it off with him because he has lost two wives to cancer and i don't want him hurt again. right now my health is still halfway decent, and we can go out and have a great time together. but all that's going to happen is we will grow closer and closer, and he's the one who will lose in the long run. he doesn't deserve to lose someone else he loves. it's not fair. is it wrong to keep dating him, or should i break it off while we still have good memories? -- slowly dying in texas", "year": 2009}, {"title": "Receptionist Won't Let Woman Outgrow Nickname of Her Youth", "text": "i am a 48-year-old woman who was known by my nickname, \"pudge,\" while i was in high school because so many other girls had the same common name. after high school, i went back to my given name, and i have carefully told all my old friends that, while my nickname was cute for a 15-year-old, it no longer suited me. most of them have made the change out of respect for me. what should i tell my doctor's receptionist, who did not know me before, but insists on using my nickname? i have told her i prefer my given name, but she refuses to use it. i don't want to hurt her feelings, but i think she should address me as i introduced myself. i see this doctor four times a year, so i see her often. she also uses the nickname on mail sent to my home. the best she has ever done is to preface it with \"mr", "year": 2006}, {"title": "Nanny Grows Tired of Playing Hide and Seek With Single Dad", "text": "my husband, \"donald,\" is working out of state. last week when i called him on his cell phone, someone picked up and said nothing -- but didn't disconnect. so for the next hour, i listened to my husband in a bar with another woman. i heard laughing, talking and glasses clinking. i heard them leave together to have dinner. then the battery died. i am hurt to the core. donald swears nothing happened, that she was just his ride. i'm trying hard to believe him, but when i question him further, he becomes upset and defensive. his answers -- or lack of them -- have destroyed my heart and soul. why can't donald say the right things to take my hurt away? why doesn't he understand? abby, am i wrong to be so upset? -- disconnected in deer park", "year": 2004}, {"title": "Compulsive Womanizer Has Now Expanded His Options", "text": "i have two teenage stepsons living with me and their mother. the older boy, \"jake,\" who is 16, wants his mother to take him and his brother out once a week or so to be alone with her, while excluding me and my daughter. jake is very shy and an introvert. i feel that this is contrary to the common good and will promote a lack of trust in the home. however, i love my girlfriend very much and will do anything to keep her happy. am i wrong for feeling betrayed over this? -- stepfather in massachusetts", "year": 2006}, {"title": "Fiance Comes Clean About Drug Use One Month Before Wedding", "text": "my fiance, \"doug,\" just revealed to me that for the past six months he's been using drugs. we've been together almost four years and our wedding is scheduled for next month. we are both in our early 20s. doug confessed that he has been using money we set aside for bills to buy drugs. he said he has also stolen money from our best friend for the same purpose. he came to me on his own to tell me all this. doug has always been a sweet, caring guy. i love him with all my heart, but i've lost my trust in him. now i don't know what to do. i can hardly believe this is happening. i still want to marry him, but don't want to marry someone i don't trust. what should i do? i need an answer in a hurry. -- hurt and confused in florida", "year": 2003}, {"title": "Niece's College Plans Shouldn't Include Rooming With Grandparents", "text": "my folks are in their mid-70s and have health problems. my oldest niece, \"riley,\" will graduate from high school next spring and is considering going to a college near them. my parents recently told me that my brother is suggesting riley move in with them. the girl has some behavioral issues and is in counseling. she's not an easy, happy or normal kid. my parents are extremely uncomfortable with the idea, but have not said anything to my brother. i think they are afraid of a fight or causing hurt feelings. he is in denial about his daughter's problems. i'm concerned about my parents. at their age, i don't think it's fair to expect them to have another teenager in their home, much less one with issues. is it my place to say something, and if so, what do i say? -- looking out for mom and dad", "year": 2014}, {"title": "Church Ladies Seem Eager to Break a Commandment", "text": "i have been in a relationship with \"sid\" for two years, but things haven't been good between us for the last eight months. we called off our wedding but are still dating. i care for sid, but sometimes i feel we have reached a dead end. i recently met another man, \"larry,\" who wants to date me. larry is very nice and says he'll understand if we don't date right now -- he's willing to wait. abby, i feel i should be by myself for a while. i haven't told sid anything yet. i don't want to hurt him. what should i do? -- confused in south carolina", "year": 2001}, {"title": "Heartbroken Mom Wants More Than Sex With Kids' Father", "text": "my heart is broken. i don't know how to fix it, and sometimes i want to kill myself. i'm in love with my children's father and he knows it. \"brad\" comes over to have sex with me, but we're not together. he tells me he's single, but i know he's with someone else. i want him to be honest -- give me that much respect -- because i have two kids by him. brad is the only person i'm having sex with. i told him i'm getting too old to play games. i'm trying to get on with my life, but still we have sex. when do i say enough is enough? i tell brad i need to drop the kids off, and he tells me no. but i need some alone time, too. if i had known our relationship would turn out like this, i would never have gotten involved with him. i love him with all my heart. please tell me what to do. -- heartsick i", "year": 2009}, {"title": "TOILET SEAT FLAP COMES DOWN TO COURTESY", "text": "this is in response to the woman who was upset because her husband of 12 years won't leave the toilet seat down for her. every time i've read this complaint in your column, i've meant to write to give the man's side, but prior letters haven't frosted my cookie like hers did. so here i am. pray tell, where is it written that women have the god-given right to the toilet seat in the position they prefer? if men are expected to position the seat for their spouse's convenience, why is it different for women? consideration works both ways, abby. well, i'm glad i got that off my chest. you may not agree with me, but you have always been fair in printing both sides of a story. for that, i thank you. you may use my name. bob ruo, palm springs, calif.", "year": 1995}, {"title": "Hard Sell Is Hard To Take At Shopping Malls", "text": "i have a problem dealing with shopping mall kiosk operators. many of them are outright obnoxious. they block your way and insist that you listen to their pitch or try their product. i find i have to avoid eye contact with them. they might say something nice as i walk by, but if i answer, it is a guaranteed lead-in to a sales pitch. i feel bad for not replying, but it's the only way. i know they are trying to make a living, but i can see their product as i walk by. if it's something i'm interested in, i'll stop and ask. otherwise, i think they should respect my privacy. am i wrong for feeling this way? -- bothered in tempe, ariz.", "year": 2014}, {"title": "TRUTH IS BEST IN UNWED DAUGHTER'S INSEMINATION", "text": "a friend of mine asked if she could borrow my wedding dress for her wedding because she wanted to keep her expenses to a minimum. i told her she could wear it with pleasure, and i carried it to her. she asked me to be her matron of honor and i was thrilled, until she told me that the dress she had chosen for her attendants would cost me $200! when i told her that $200 was a little too steep for my pocketbook, she became upset. to make a long story short, she eliminated me from the wedding party entirely, and i was so hurt i did not attend her wedding. abby, shouldn't the bride consult with her attendants concerning the price of the gowns the attendants are expected to pay for? and do you think i was wrong to refuse to go into debt to buy the dress she selected? by the way, she wore my wedd", "year": 1985}, {"title": "Cabbie's wife thinks she smells tall story", "text": "my sister, who is divorced, recently took a full-time job. she has an 8-year-old daughter, cissy. she refuses to get a baby sitter for cissy, saying the child is old enough to take care of herself for the three hours after school until my sister gets home. i am really worried about my niece. she is a quiet child and i am concerned about the responsibility this thrusts on her right after losing her father (a year ago). my mother has threatened to report the situation to the child services department in our town. sis thinks we're being silly and says she can't afford a sitter even if she felt one was needed. mother and i both work, so we can't volunteer our services. i don't want a family fight, but i feel the welfare of the child is at stake. what should we do? concerned", "year": 1990}, {"title": "Diary Opens Door to Dialogue Between Mother and Daughter", "text": "i'm a 16-year-old girl who accidentally left my diary on the counter and my mother read it. when she told me, i was disappointed and hurt. to me, a diary is a place i can escape to and feel comfortable just being me. she now knows i struggle with depression and have done things i'm not proud of. i was angry and expected an apology because it was a violation of my privacy. she claims she had the right to read it because i left it on the counter, and if i didn't want her to see it, i shouldn't have left it there. regardless of where my diary was, i don't feel she had the right to go through it because it's not hers. i told her i want an apology and i am willing to rebuild that trust. my mom said there is no reason to rebuild it or to apologize, and she did nothing wrong. am i wrong for wanti", "year": 2012}, {"title": "Woman Fears Being Watched by Ghosts of Her Loved Ones", "text": "i have a question regarding gift giving. if you receive a gift of clothing (with a receipt) from someone and the garment doesn't fit, is it your responsibility to exchange it, or should you return it to the gift-giver, explain that it's the wrong size and ask the person to return it? i gave my sister an outfit that didn't fit her. she immediately gave the gift back and asked me to return it. -- lori in fountain valley, calif.", "year": 2010}, {"title": "Grandparents' Early Memories Are Cherished Family History", "text": "i'm a fairly intelligent 45-year-old woman. after being single for four years, i began dating a man my age with whom i share many interests. early on, we had a few fights -- possibly because we were both hurt in our previous relationships and were having a hard time adjusting to and trusting a new person. things have settled down now. most of our time is spent together even though we live an hour apart, and we're considered a couple by our friends. i enjoy the time we spend together, but i keep remembering our early fights and i worry about repeats. i think because of our pasts we'll date for a long time before either of us considers moving in or making serious commitments. my question is, how can you know if you're on the right path? -- a little skittish in canada", "year": 2012}, {"title": "LIVE-IN MAY HAVE TO SPEND TIME TO EARN FRIENDS", "text": "my wife has a degree in the medical field from a large, respected university, yet she thinks it's perfectly all right to allow our dog to drink out of the toilet bowl. the dog is even allowed to drink out of our swimming pool. she says it won't hurt him. this makes no sense at all to me. the pool man puts chemicals into the pool to kill the algae, so wouldn't that be harmful to our dog? please hurry your answer. we are having words about this. this is a second marriage for both of us, and i can't afford another divorce. dog tired", "year": 1987}, {"title": "Move To London Hasn't Panned Out For Half Of Two-Career Couple", "text": "my husband and i recently quit our jobs and moved to london from new york. being a freelancer and having lived here before, he's never had trouble finding work. but i have just changed careers, and i'm finding it hard to earn a consistent paycheck here. despite his constant assurances that he is happy supporting both of us right now, i can't shake feeling guilty. i have never felt right living on someone else's dime -- not even my parents' while i was growing up. should i man up and find a job i don't exactly love to better contribute, or \"keep on truckin'\" without guilt with hopes of getting there? -- guilty in london", "year": 2015}]''')

random.shuffle(DEAR_ABBY)
print(f"  Dear Abby total: {len(DEAR_ABBY)} letters ({min(d['year'] for d in DEAR_ABBY)}-{max(d['year'] for d in DEAR_ABBY)})")
print(f"  (for T2-T5: structural tests — no verdict labels needed)\n")

DIMENSIONS = [
    "consequences", "rights", "fairness", "autonomy",
    "trust", "social_impact", "identity"
]

DIM_DESCRIPTIONS = {
    "consequences": "Who was harmed? Who benefited? How severely?",
    "rights": "Whose rights were respected or violated?",
    "fairness": "Was this fair to all parties? Was there reciprocity?",
    "autonomy": "Was anyone's freedom of choice restricted?",
    "trust": "Was trust maintained or broken?",
    "social_impact": "How did this affect relationships and community?",
    "identity": "What does this say about the person's character?",
}


# ═══════════════════════════════════════════════════════════════════════
# 2. STRUCTURED OUTPUT SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DimensionalVerdict:
    verdict: str
    confidence: float
    reasoning: str

@dataclass
class FullDimensionalAnalysis:
    consequences_score: float
    rights_score: float
    fairness_score: float
    autonomy_score: float
    trust_score: float
    social_impact_score: float
    identity_score: float
    verdict: str
    confidence: float
    reasoning: str

@dataclass
class InvarianceJudgment:
    verdict: str
    confidence: float
    severity: float
    reasoning: str

@dataclass
class OrderedJudgment:
    verdict: str
    confidence: float
    reasoning: str

@dataclass
class HarmAssessment:
    consequences_harm: float
    rights_harm: float
    fairness_harm: float
    autonomy_harm: float
    trust_harm: float
    social_impact_harm: float
    identity_harm: float
    total_harm: float
    verdict: str
    reasoning: str


# ═══════════════════════════════════════════════════════════════════════
# 3. HELPERS
# ═══════════════════════════════════════════════════════════════════════

def normalize_verdict(raw):
    raw = str(raw).strip().upper().replace("'", "").replace(".", "")
    for label in ["NTA", "YTA", "ESH", "NAH"]:
        if label in raw:
            return label
    return "UNK"

def mean(xs):
    xs = list(xs)
    return sum(xs) / max(len(xs), 1)

def stdev(xs):
    xs = list(xs)
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5

def pearson_r(xs, ys):
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    mx, my = mean(xs[:n]), mean(ys[:n])
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = sum((xs[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ys[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx < 1e-10 or dy < 1e-10:
        return 0.0
    return num / (dx * dy)

def gini(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0 or sum(xs) < 1e-10:
        return 0.0
    cum = sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs))
    return cum / (n * sum(xs))

def sigma_level(p, n):
    """How many σ is proportion p from 0, given n trials?"""
    if p <= 0 or n <= 0:
        return 0.0
    se = (p * (1 - p) / n) ** 0.5
    return p / max(se, 1e-10)

def clamp(v, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = (lo + hi) / 2
    return max(lo, min(hi, v))


def call_llm(llm, chat_id, prompt_text, schema=None):
    """Thread-safe LLM call with adaptive backoff."""
    try:
        with kbench.chats.new(chat_id):
            result = llm.prompt(prompt_text, schema=schema) if schema else llm.prompt(prompt_text)
        _pool.record_success()
        return result
    except Exception as e:
        _pool.record_failure()
        raise


def call_llm_2step(llm, gen_id, judge_id, gen_prompt, judge_prompt_fn, schema):
    """Thread-safe 2-step with adaptive backoff."""
    try:
        with kbench.chats.new(gen_id):
            generated = llm.prompt(gen_prompt)
        _pool.record_success()
        with kbench.chats.new(judge_id):
            result = llm.prompt(judge_prompt_fn(generated), schema=schema)
        _pool.record_success()
        return result
    except Exception as e:
        _pool.record_failure()
        raise


def parallel_calls(tasks):
    """Run tasks with adaptive concurrency. tasks = list of (fn, args, key)."""
    results = {}
    remaining = list(tasks)
    while remaining:
        batch_size = min(_pool.n, len(remaining))
        batch = remaining[:batch_size]
        remaining = remaining[batch_size:]
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {}
            for fn, args, key in batch:
                futures[pool.submit(fn, *args)] = key
            for f in as_completed(futures):
                key = futures[f]
                try:
                    results[key] = f.result()
                except Exception as e:
                    results[key] = None
                    print(f"    WARN: {key} failed: {e}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# 4. PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def base_scenario_text(s):
    return f"Title: {s['title']}\n\n{s['text']}"

VERDICT_OPTIONS = (
    "YTA = You're The Asshole (poster is wrong)\n"
    "NTA = Not The Asshole (poster is right)\n"
    "ESH = Everyone Sucks Here (both sides are wrong)\n"
    "NAH = No Assholes Here (nobody is wrong)"
)

def prompt_subset_dims(s, dims):
    dim_block = "\n".join(f"  - {d.replace('_', ' ').title()}: {DIM_DESCRIPTIONS[d]}" for d in dims)
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n"
        f"Evaluate ONLY using these moral dimensions (ignore all others):\n"
        f"{dim_block}\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Rate confidence 0-10."
    )

def prompt_full_dims(s):
    dim_block = "\n".join(f"  - {d.replace('_', ' ').title()}: {DIM_DESCRIPTIONS[d]}" for d in DIMENSIONS)
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n"
        f"Score each moral dimension from -5 (severe harm) to +5 (clear benefit):\n"
        f"{dim_block}\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Give scores for each dimension, then your verdict:\n{VERDICT_OPTIONS}\n"
        f"Rate confidence 0-10."
    )

def prompt_invariance(s):
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Rate severity 0-10 and confidence 0-10."
    )

def prompt_gender_swap(s):
    return (
        f"Rewrite the following scenario, swapping all genders "
        f"(male→female, female→male, he→she, etc.). "
        f"Keep ALL facts, actions, and moral content identical. "
        f"Only change gender references.\n\n"
        f"{base_scenario_text(s)}"
    )

def prompt_reframe_neutral(s):
    return (
        f"Rewrite the following scenario changing all names and "
        f"shifting the cultural context (e.g., family dinner→work event). "
        f"Keep ALL moral facts and actions identical. "
        f"Only change surface details.\n\n"
        f"{base_scenario_text(s)}"
    )

def prompt_reorder_victim_first(s):
    return (
        f"Rewrite the following scenario so it is told from the "
        f"affected party's (victim's) perspective first, then the poster's actions, "
        f"then the broader context. Keep ALL facts identical.\n\n"
        f"{base_scenario_text(s)}"
    )

def prompt_reorder_context_first(s):
    return (
        f"Rewrite the following scenario starting with the broader context "
        f"and background, then the poster's actions, then the impact on others. "
        f"Keep ALL facts identical.\n\n"
        f"{base_scenario_text(s)}"
    )

def prompt_ordered_dims(s, dim_a, dim_b):
    desc_a = DIM_DESCRIPTIONS[dim_a]
    desc_b = DIM_DESCRIPTIONS[dim_b]
    name_a = dim_a.replace("_", " ").title()
    name_b = dim_b.replace("_", " ").title()
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n"
        f"Evaluate in exactly two steps:\n\n"
        f"STEP 1: Consider ONLY {name_a}: {desc_a}\n"
        f"Form your initial assessment.\n\n"
        f"STEP 2: Now, incorporating your {name_a} assessment, "
        f"evaluate {name_b}: {desc_b}\n"
        f"Give your FINAL verdict based on both steps.\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Rate confidence 0-10."
    )

def prompt_euphemistic(s):
    return (
        f"Rewrite the following scenario using softer, more neutral, "
        f"euphemistic language. Minimize emotional impact. "
        f"Keep ALL facts and actions identical.\n\n"
        f"{base_scenario_text(s)}"
    )

def prompt_dramatic(s):
    return (
        f"Rewrite the following scenario in emotionally charged, "
        f"dramatic language. Maximize emotional impact. "
        f"Keep ALL facts and actions identical.\n\n"
        f"{base_scenario_text(s)}"
    )

def prompt_harm(text):
    return (
        f"You are assessing harm in a post from Reddit's r/AmITheAsshole.\n"
        f"Rate harm on each dimension from 0 (no harm) to 10 (severe harm):\n"
        f"  - Consequences, Rights, Fairness, Autonomy, Trust, Social Impact, Identity\n\n"
        f"{text}\n\n"
        f"Also give total harm (0-70) and your verdict:\n{VERDICT_OPTIONS}"
    )

def judge_prompt(text):
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n\n"
        f"{text}\n\n"
        f"Verdict:\n{VERDICT_OPTIONS}\nRate severity 0-10 and confidence 0-10."
    )


# ═══════════════════════════════════════════════════════════════════════
# T1: STRUCTURAL FUZZING OF MORAL DIMENSIONS
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="t1_structural_fuzzing")
def t1_structural_fuzzing(llm):
    print("\n[T1] STRUCTURAL FUZZING OF MORAL DIMENSIONS")
    print("  Ablating dimensions to find sensitivity profile")
    print("-" * 60)

    scenarios = AITA_SCENARIOS[:10]  # Budget: 10 scenarios × 15 calls = 150
    solo_accuracy = {d: {"correct": 0, "total": 0} for d in DIMENSIONS}
    loo_flips = {d: 0 for d in DIMENSIONS}
    loo_total = {d: 0 for d in DIMENSIONS}
    baseline_correct = 0
    baseline_total = 0
    _lock = threading.Lock()

    for si, s in enumerate(scenarios):
        actual = s["verdict"]

        # Baseline first (needed for LOO comparison)
        base = call_llm(llm, f"t1_base_{si}", prompt_full_dims(s), FullDimensionalAnalysis)
        base_pred = normalize_verdict(base.verdict)
        if base_pred == actual:
            baseline_correct += 1
        baseline_total += 1

        # Solo + LOO in parallel (14 calls)
        futures = {}
        with ThreadPoolExecutor(max_workers=_pool.n) as pool:
            for d in DIMENSIONS:
                f = pool.submit(call_llm, llm, f"t1_solo_{si}_{d}",
                                prompt_subset_dims(s, [d]), DimensionalVerdict)
                futures[f] = ("solo", d)

                remaining = [x for x in DIMENSIONS if x != d]
                f2 = pool.submit(call_llm, llm, f"t1_loo_{si}_{d}",
                                 prompt_subset_dims(s, remaining), DimensionalVerdict)
                futures[f2] = ("loo", d)

            for f in as_completed(futures):
                kind, d = futures[f]
                try:
                    j = f.result()
                    pred = normalize_verdict(j.verdict)
                    with _lock:
                        if kind == "solo":
                            solo_accuracy[d]["total"] += 1
                            if pred == actual:
                                solo_accuracy[d]["correct"] += 1
                        else:
                            loo_total[d] += 1
                            if pred != base_pred:
                                loo_flips[d] += 1
                except Exception as e:
                    print(f"    WARN: {kind} {d} failed: {e}")

        n = si + 1
        print(f"  [{n}/{len(scenarios)}] baseline={'correct' if base_pred == actual else 'MISS'}")

    solo_acc = {d: solo_accuracy[d]["correct"] / max(solo_accuracy[d]["total"], 1) for d in DIMENSIONS}
    loo_flip_rate = {d: loo_flips[d] / max(loo_total[d], 1) for d in DIMENSIONS}
    ranked = sorted(DIMENSIONS, key=lambda d: loo_flip_rate[d], reverse=True)
    profile_sharpness = gini([loo_flip_rate[d] for d in DIMENSIONS])

    print(f"\n  RESULTS:")
    print(f"  Baseline accuracy: {baseline_correct}/{baseline_total}")
    print(f"  {'Dimension':<20} {'Solo Acc':>10} {'LOO Flip Rate':>15} {'Rank':>6}")
    print(f"  {'-'*51}")
    for rank, d in enumerate(ranked, 1):
        name = d.replace("_", " ").title()
        print(f"  {name:<20} {solo_acc[d]:>9.0%} {loo_flip_rate[d]:>14.0%} {rank:>6}")
    print(f"  Profile sharpness (Gini): {profile_sharpness:.3f}")

    _results_store["T1_fuzzing"] = {
        "solo_accuracy": solo_acc,
        "loo_flip_rate": loo_flip_rate,
        "importance_ranking": ranked,
        "profile_sharpness": profile_sharpness,
        "baseline_accuracy": baseline_correct / max(baseline_total, 1),
        "score": profile_sharpness,
    }


# ═══════════════════════════════════════════════════════════════════════
# T2: BOND INVARIANCE PRINCIPLE
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="t2_invariance")
def t2_invariance(llm):
    print("\n[T2] BOND INVARIANCE PRINCIPLE")
    print("  Testing verdict stability under re-description")
    print("-" * 60)

    scenarios = DEAR_ABBY[:20]  # Budget: 20 × 5 calls = 100
    gender_flips = 0
    reframe_flips = 0
    severity_drifts = []
    total = 0

    for si, s in enumerate(scenarios):
        # Baseline
        base = call_llm(llm, f"t2_base_{si}", prompt_invariance(s), InvarianceJudgment)
        base_v = normalize_verdict(base.verdict)
        base_sev = clamp(base.severity, 0, 10)

        # Gender swap + reframe in parallel (2-step each)
        with ThreadPoolExecutor(max_workers=min(_pool.n, 2)) as pool:
            f_gender = pool.submit(
                call_llm_2step, llm,
                f"t2_gswap_{si}", f"t2_gjudge_{si}",
                prompt_gender_swap(s), judge_prompt, InvarianceJudgment
            )
            f_reframe = pool.submit(
                call_llm_2step, llm,
                f"t2_rswap_{si}", f"t2_rjudge_{si}",
                prompt_reframe_neutral(s), judge_prompt, InvarianceJudgment
            )

            gj = f_gender.result()
            rj = f_reframe.result()

        g_v = normalize_verdict(gj.verdict)
        g_sev = clamp(gj.severity, 0, 10)
        r_v = normalize_verdict(rj.verdict)
        r_sev = clamp(rj.severity, 0, 10)

        total += 1
        if g_v != base_v:
            gender_flips += 1
        if r_v != base_v:
            reframe_flips += 1
        severity_drifts.append(abs(g_sev - base_sev))
        severity_drifts.append(abs(r_sev - base_sev))

        n = si + 1
        flipped = "GENDER" if g_v != base_v else ""
        flipped += " REFRAME" if r_v != base_v else ""
        if n % 5 == 0 or flipped:
            print(f"  [{n}/{len(scenarios)}] base={base_v} gender={g_v} reframe={r_v} {flipped}")

    gender_rate = gender_flips / max(total, 1)
    reframe_rate = reframe_flips / max(total, 1)
    overall_violation = (gender_flips + reframe_flips) / max(2 * total, 1)
    avg_sev_drift = mean(severity_drifts)

    sig_g = sigma_level(gender_rate, total)
    sig_r = sigma_level(reframe_rate, total)
    sig_all = sigma_level(overall_violation, 2 * total)

    print(f"\n  RESULTS:")
    print(f"  Gender-swap violation rate: {gender_flips}/{total} ({gender_rate:.0%}) [{sig_g:.1f}σ]")
    print(f"  Reframe violation rate: {reframe_flips}/{total} ({reframe_rate:.0%}) [{sig_r:.1f}σ]")
    print(f"  Overall invariance violation: {overall_violation:.0%} [{sig_all:.1f}σ]")
    print(f"  Mean severity drift: {avg_sev_drift:.2f}")
    print(f"  6σ threshold met: {'YES' if min(sig_g, sig_r) >= 6 else 'NO'} (need both ≥6.0)")

    _results_store["T2_invariance"] = {
        "gender_violation_rate": gender_rate,
        "reframe_violation_rate": reframe_rate,
        "overall_violation_rate": overall_violation,
        "severity_drift": avg_sev_drift,
        "score": 1.0 - overall_violation,
    }


# ═══════════════════════════════════════════════════════════════════════
# T3: HOLONOMY / PATH-DEPENDENCE
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="t3_holonomy")
def t3_holonomy(llm):
    print("\n[T3] HOLONOMY — PATH-DEPENDENCE OF MORAL JUDGMENT")
    print("  Same facts, different presentation order")
    print("-" * 60)

    scenarios = DEAR_ABBY[:20]  # Budget: 20 × 5 calls = 100
    path_dependent = 0
    total = 0
    order_verdicts = []

    for si, s in enumerate(scenarios):
        # Original + two reorders in parallel
        with ThreadPoolExecutor(max_workers=min(_pool.n, 3)) as pool:
            f_orig = pool.submit(call_llm, llm, f"t3_orig_{si}",
                                 prompt_invariance(s), InvarianceJudgment)
            f_victim = pool.submit(
                call_llm_2step, llm,
                f"t3_vgen_{si}", f"t3_vjudge_{si}",
                prompt_reorder_victim_first(s), judge_prompt, InvarianceJudgment
            )
            f_context = pool.submit(
                call_llm_2step, llm,
                f"t3_cgen_{si}", f"t3_cjudge_{si}",
                prompt_reorder_context_first(s), judge_prompt, InvarianceJudgment
            )

            orig = f_orig.result()
            vj = f_victim.result()
            cj = f_context.result()

        v_orig = normalize_verdict(orig.verdict)
        v_victim = normalize_verdict(vj.verdict)
        v_context = normalize_verdict(cj.verdict)

        total += 1
        verdicts = {v_orig, v_victim, v_context}
        if len(verdicts) > 1:
            path_dependent += 1
        order_verdicts.append((v_orig, v_victim, v_context))

        n = si + 1
        marker = " PATH-DEP" if len(verdicts) > 1 else ""
        if n % 5 == 0 or marker:
            print(f"  [{n}/{len(scenarios)}] orig={v_orig} victim={v_victim} "
                  f"context={v_context}{marker}")

    holonomy_rate = path_dependent / max(total, 1)
    all_agree = sum(1 for o, v, c in order_verdicts if o == v == c)
    two_agree = sum(1 for o, v, c in order_verdicts if len({o, v, c}) == 2)
    none_agree = sum(1 for o, v, c in order_verdicts if len({o, v, c}) == 3)

    sig_h = sigma_level(holonomy_rate, total)

    print(f"\n  RESULTS:")
    print(f"  Path-dependent scenarios: {path_dependent}/{total} ({holonomy_rate:.0%}) [{sig_h:.1f}σ]")
    print(f"  All 3 agree: {all_agree}/{total}")
    print(f"  2 of 3 agree: {two_agree}/{total}")
    print(f"  All different: {none_agree}/{total}")
    print(f"  6σ threshold met: {'YES' if sig_h >= 6 else 'NO'}")

    _results_store["T3_holonomy"] = {
        "holonomy_rate": holonomy_rate,
        "all_agree": all_agree / max(total, 1),
        "two_agree": two_agree / max(total, 1),
        "none_agree": none_agree / max(total, 1),
        "score": 1.0 - holonomy_rate,
    }


# ═══════════════════════════════════════════════════════════════════════
# T4: CONTRACTION ORDER — NON-COMMUTATIVITY
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="t4_contraction_order")
def t4_contraction_order(llm):
    print("\n[T4] CONTRACTION ORDER — NON-COMMUTATIVITY")
    print("  Does evaluation order change the verdict?")
    print("-" * 60)

    scenarios = DEAR_ABBY[:15]  # Budget: 15 × 6 calls = 90
    DIM_PAIRS = [
        ("consequences", "fairness"),
        ("rights", "trust"),
        ("autonomy", "identity"),
    ]

    pair_flips = {f"{a},{b}": 0 for a, b in DIM_PAIRS}
    pair_total = {f"{a},{b}": 0 for a, b in DIM_PAIRS}
    total_flips = 0
    total_tests = 0
    _lock = threading.Lock()

    for si, s in enumerate(scenarios):
        # All 6 calls (3 pairs × 2 orders) in parallel
        futures = {}
        with ThreadPoolExecutor(max_workers=min(_pool.n, 6)) as pool:
            for dim_a, dim_b in DIM_PAIRS:
                f_ab = pool.submit(call_llm, llm, f"t4_{si}_{dim_a}_{dim_b}",
                                   prompt_ordered_dims(s, dim_a, dim_b), OrderedJudgment)
                f_ba = pool.submit(call_llm, llm, f"t4_{si}_{dim_b}_{dim_a}",
                                   prompt_ordered_dims(s, dim_b, dim_a), OrderedJudgment)
                futures[(dim_a, dim_b)] = (f_ab, f_ba)

            for (dim_a, dim_b), (f_ab, f_ba) in futures.items():
                pair_key = f"{dim_a},{dim_b}"
                try:
                    v_ab = normalize_verdict(f_ab.result().verdict)
                    v_ba = normalize_verdict(f_ba.result().verdict)
                    with _lock:
                        pair_total[pair_key] += 1
                        total_tests += 1
                        if v_ab != v_ba:
                            pair_flips[pair_key] += 1
                            total_flips += 1
                except Exception as e:
                    print(f"    WARN: {pair_key} failed: {e}")

        n = si + 1
        if n % 5 == 0:
            running = total_flips / max(total_tests, 1)
            print(f"  [{n}/{len(scenarios)}] non-commutativity rate: {running:.0%}")

    noncomm_rate = total_flips / max(total_tests, 1)
    sig_nc = sigma_level(noncomm_rate, total_tests)

    print(f"\n  RESULTS:")
    print(f"  Overall non-commutativity: {total_flips}/{total_tests} ({noncomm_rate:.0%}) [{sig_nc:.1f}σ]")
    print(f"  {'Pair':<35} {'Flips':>8} {'Rate':>8} {'σ':>6}")
    print(f"  {'-'*57}")
    pair_rates = {}
    for a, b in DIM_PAIRS:
        key = f"{a},{b}"
        na = a.replace("_", " ").title()
        nb = b.replace("_", " ").title()
        rate = pair_flips[key] / max(pair_total[key], 1)
        pair_rates[key] = rate
        sig_p = sigma_level(rate, pair_total[key])
        print(f"  {na} <-> {nb:<20} {pair_flips[key]:>8} {rate:>7.0%} {sig_p:>5.1f}σ")
    print(f"  6σ threshold met: {'YES' if sig_nc >= 6 else 'NO'}")

    _results_store["T4_contraction"] = {
        "noncommutativity_rate": noncomm_rate,
        "pair_rates": pair_rates,
        "score": 1.0 - noncomm_rate,
    }


# ═══════════════════════════════════════════════════════════════════════
# T5: CONSERVATION OF HARM
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="t5_conservation")
def t5_conservation(llm):
    print("\n[T5] CONSERVATION OF HARM")
    print("  Does reframing change perceived total harm?")
    print("-" * 60)

    scenarios = DEAR_ABBY[:12]  # Budget: 12 × 5 calls = 60
    base_harms = []
    euphemistic_harms = []
    dramatic_harms = []
    verdict_flips_euph = 0
    verdict_flips_dram = 0
    total = 0

    for si, s in enumerate(scenarios):
        # Baseline + euphemistic + dramatic in parallel
        with ThreadPoolExecutor(max_workers=min(_pool.n, 3)) as pool:
            f_base = pool.submit(call_llm, llm, f"t5_base_{si}",
                                 prompt_harm(base_scenario_text(s)), HarmAssessment)
            f_euph = pool.submit(
                call_llm_2step, llm,
                f"t5_euph_gen_{si}", f"t5_euph_judge_{si}",
                prompt_euphemistic(s), prompt_harm, HarmAssessment
            )
            f_dram = pool.submit(
                call_llm_2step, llm,
                f"t5_dram_gen_{si}", f"t5_dram_judge_{si}",
                prompt_dramatic(s), prompt_harm, HarmAssessment
            )

            base = f_base.result()
            euph = f_euph.result()
            dram = f_dram.result()

        base_total = clamp(base.total_harm, 0, 70)
        base_v = normalize_verdict(base.verdict)
        base_harms.append(base_total)

        euph_total = clamp(euph.total_harm, 0, 70)
        euph_v = normalize_verdict(euph.verdict)
        euphemistic_harms.append(euph_total)

        dram_total = clamp(dram.total_harm, 0, 70)
        dram_v = normalize_verdict(dram.verdict)
        dramatic_harms.append(dram_total)

        total += 1
        if euph_v != base_v:
            verdict_flips_euph += 1
        if dram_v != base_v:
            verdict_flips_dram += 1

        n = si + 1
        drift_e = euph_total - base_total
        drift_d = dram_total - base_total
        if n % 4 == 0:
            print(f"  [{n}/{len(scenarios)}] base={base_total:.0f} euph={euph_total:.0f}({drift_e:+.0f}) "
                  f"dram={dram_total:.0f}({drift_d:+.0f})")

    r_euph = pearson_r(base_harms, euphemistic_harms)
    r_dram = pearson_r(base_harms, dramatic_harms)
    mean_drift_euph = mean([e - b for b, e in zip(base_harms, euphemistic_harms)])
    mean_drift_dram = mean([d - b for b, d in zip(base_harms, dramatic_harms)])
    conservation = (r_euph + r_dram) / 2

    sig_e = sigma_level(verdict_flips_euph / max(total, 1), total)
    sig_d = sigma_level(verdict_flips_dram / max(total, 1), total)

    print(f"\n  RESULTS:")
    print(f"  Base-Euphemistic correlation: r={r_euph:.3f}, mean drift={mean_drift_euph:+.1f}")
    print(f"  Base-Dramatic correlation: r={r_dram:.3f}, mean drift={mean_drift_dram:+.1f}")
    print(f"  Conservation score: {conservation:.3f}")
    print(f"  Verdict flips (euphemistic): {verdict_flips_euph}/{total} [{sig_e:.1f}σ]")
    print(f"  Verdict flips (dramatic): {verdict_flips_dram}/{total} [{sig_d:.1f}σ]")
    print(f"  (Conservation measured by correlation, not proportion — 6σ N/A)")

    _results_store["T5_conservation"] = {
        "r_euphemistic": r_euph,
        "r_dramatic": r_dram,
        "mean_drift_euphemistic": mean_drift_euph,
        "mean_drift_dramatic": mean_drift_dram,
        "conservation": conservation,
        "verdict_flips_euphemistic": verdict_flips_euph / max(total, 1),
        "verdict_flips_dramatic": verdict_flips_dram / max(total, 1),
        "score": max(0, conservation),
    }


# ═══════════════════════════════════════════════════════════════════════
# MULTI-MODEL EXECUTION
# ═══════════════════════════════════════════════════════════════════════

MODELS_TO_TEST = [
    "google/gemini-2.0-flash",       # ~$0.014/call — baseline
    "google/gemini-2.5-flash",       # ~$0.014/call — current gen
    "google/gemini-2.5-pro",         # ~$0.014/call — strongest
    "google/gemini-3-flash-preview", # ~$0.014/call — next gen
]
# Budget: 4 models × ~500 calls × $0.014 = ~$28 (within $50 quota)

print(f"\n[2/8] Running 5 geometric tests across {len(MODELS_TO_TEST)} models")
for m in MODELS_TO_TEST:
    print(f"  - {m}")
print()

all_results = {}

for model_name in MODELS_TO_TEST:
    print(f"\n{'#'*70}")
    print(f"# MODEL: {model_name}")
    print(f"{'#'*70}")

    model_results = {}
    try:
        llm = kbench.llms[model_name]
        _results_store.clear()  # reset for each model

        for test_fn, test_name in [
            (t1_structural_fuzzing, "T1_fuzzing"),
            (t2_invariance, "T2_invariance"),
            (t3_holonomy, "T3_holonomy"),
            (t4_contraction_order, "T4_contraction"),
            (t5_conservation, "T5_conservation"),
        ]:
            try:
                test_fn.run(llm=llm)
                model_results[test_name] = _results_store.get(test_name, {"score": 0.0})
            except Exception as e:
                print(f"  ERROR in {test_name}: {e}")
                model_results[test_name] = {"error": str(e), "score": 0.0}

    except Exception as e:
        print(f"  ERROR loading model {model_name}: {e}")
        model_results = {f"T{i}": {"error": str(e), "score": 0.0} for i in range(1, 6)}

    all_results[model_name] = model_results


# ═══════════════════════════════════════════════════════════════════════
# CROSS-MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*70}")
print(f"CROSS-MODEL COMPARISON — FIVE GEOMETRIC TESTS")
print(f"{'#'*70}")
print()

WEIGHTS = {
    "T1_fuzzing": 0.25,
    "T2_invariance": 0.20,
    "T3_holonomy": 0.20,
    "T4_contraction": 0.15,
    "T5_conservation": 0.20,
}

header = f"  {'Model':<30} {'T1:Fuzz':>8} {'T2:BIP':>8} {'T3:Holo':>8} {'T4:Comm':>8} {'T5:Cons':>8} {'Compos':>8}"
print(header)
print(f"  {'─'*78}")

for model_name, results in all_results.items():
    scores = {}
    for test_key in WEIGHTS:
        r = results.get(test_key, {})
        scores[test_key] = r.get("score", 0.0)

    composite = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)

    short_name = model_name.split("/")[-1][:28]
    print(f"  {short_name:<30} "
          f"{scores['T1_fuzzing']:>7.3f} "
          f"{scores['T2_invariance']:>7.3f} "
          f"{scores['T3_holonomy']:>7.3f} "
          f"{scores['T4_contraction']:>7.3f} "
          f"{scores['T5_conservation']:>7.3f} "
          f"{composite:>7.3f}")

print()
print(f"  Weights: T1={WEIGHTS['T1_fuzzing']}, T2={WEIGHTS['T2_invariance']}, "
      f"T3={WEIGHTS['T3_holonomy']}, T4={WEIGHTS['T4_contraction']}, T5={WEIGHTS['T5_conservation']}")
print()

print("INTERPRETATION")
print("=" * 70)
print()
print("  T1 (Structural Fuzzing): Higher = sharper sensitivity profile.")
print("  T2 (Bond Invariance): Higher = more invariant under re-description.")
print("  T3 (Holonomy): Higher = less path-dependent.")
print("  T4 (Contraction Order): Higher = more commutative.")
print("  T5 (Conservation of Harm): Higher = better harm conservation.")
print()
print("  These 5 tests operationalize predictions from Geometric Ethics")
print("  (Bond, 2026) as quantitative benchmarks for social cognition.")
print("=" * 70)

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
