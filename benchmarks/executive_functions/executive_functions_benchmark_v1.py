"""Executive Functions Benchmark v1 — Dear Abby, Six Sigma Edition
Executive Functions Track | Measuring Progress Toward AGI Competition

Tests 4 executive function capabilities:
  E1. Cognitive Flexibility — can the model genuinely shift ethical frameworks?
  E2. Inhibitory Control — can the model resist emotional anchoring?
  E3. Planning / Multi-Step Reasoning — causal chain tracing and what-if analysis
  E4. Working Memory — tracking morally relevant parties at scale

All tests use Dear Abby (1985-2017) embedded dataset.
Sample sizes calibrated for 6σ significance.
Adaptive parallelism (CSMA/CA-style concurrency control).
5 frontier models across 3 providers.

Paste this ENTIRE file into ONE cell in a Kaggle Benchmark Task notebook.
Expected runtime: ~2-4 hours (adaptive, 5 models × ~450 calls each).
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
print("EXECUTIVE FUNCTIONS BENCHMARK v1 (adaptive parallelism)")
print("Four Tests of Executive Function Capabilities")
print(f"Initial parallelism: {WORKERS_INIT} (adaptive {WORKERS_MIN}-{WORKERS_MAX})")
print("=" * 70)
print()

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD DATASETS
# ═══════════════════════════════════════════════════════════════════════

t0 = time.time()

# --- Dear Abby dataset (for all tests — no labels needed) ---
print("[1/8] Loading Dear Abby scenarios (embedded)...")

DEAR_ABBY = json.loads(r'''[{"title": "Widow Realizes the Perfect Gift Is Giving Love to Others", "text": "last week, my family suffered the loss of my grandfather. he was catholic, the only catholic in our immediate family, and his funeral was held in a catholic church as he wished. when it came time to receive communion, a family friend encouraged my grandmother and the rest of the non-catholic family members to receive communion. should we have received communion out of respect for our grandfather, or was it right to stand by our own beliefs? -- grieving in virginia", "year": 2005}, {"title": "A SECOND LOOK AT ABUSIVE FIANCE", "text": "i just finished reading the letter from the 19-year-old girl who signed herself \"second thoughts in moline, ill.\" abby, i could have written that letter myself. i'm also 19, have gone with a guy for nearly three years and we plan to be married soon, but i, too, have second thoughts for the same reasons. he's jealous, doesn't trust me, accuses me of going out with other guys, and we fight a lot about stupid things. i like people and enjoy having friends, but he always wants to be with me alone. lately he has cursed me, accused me of going out on him and has hit me. then he cries, begs me to give him another chance and swears he'll never do it again, but he does. my friends tell me i'll be sorry if i marry him, but, abby, i truly love him. can you tell me why women go right on loving guys wh", "year": 1988}, {"title": "Teen Directs Increasing Anger Toward His Mother and Sister", "text": "i'm afraid something is wrong with my younger brother. he's just turned 13, and he's become violent and angry. before, he used to tease me and our sister once in a while, but now it's becoming an everyday thing. if we tease him back, he gets mad and starts yelling. he directs most of his anger at our younger sister and our mom, and he has started to push my mom. she's worried that he might hurt one of us. if she confronts him when he gets out of hand, he starts yelling, \"what? i didn't do anything!\" then later, he'll come back and try to push or kick her. i have noticed that he talks to himself, too. we don't know what to do. we hear how kids sometimes harm their families. i'm afraid for my mom and my sister, but also for my brother. i love him and want to help him. -- afraid in arizona", "year": 2016}, {"title": "Therapist Frustrated by Parents Who Make Excuses for Kids", "text": "i have been dating \"charlie\" for a year. we are going to move in together at the end of the month. charlie is thoughtful and sweet, and for the first time in my life, i feel i can be totally myself around a man. last night, i had my feet propped up on his book bag and the bag fell open. i looked down and saw a pair of black women's panties in the style that he has been badgering me to wear. next to them were two dvds with pornographic pictures on the front. i was horrified. i have trusted charlie because he really doesn't have time to cheat on me. but why would he carry around another girl's underwear? abby, i can't think straight right now. i don't want to make a mistake, and i have no idea what to do. please help. -- scared to move in new york", "year": 2003}, {"title": "Boyfriend's Dating Profile Artfully Dodges the Truth", "text": "my boyfriend has posted his profile on a dating web site in the hope of finding some new friends. i am frequently out of town on business, and he has decided that he would like to converse with \"artsy\" people during the week while i am away. he claims this web site is the only way to meet like-minded people. while i don't mind his wanting to meet people, i feel that using a dating web site is inappropriate. i read his profile; in it he indicates that he is \"single.\" (he promises he will tell the woman he meets that he is not single \"when and if the topic comes up.\") i think it's wrong to meet people based on a lie. he swears he would never cheat on me. how can i convince him that this is a form of cheating and that it's disrespectful to me? -- frustrated in new york", "year": 2004}, {"title": "Moving Child's Grave Sparks Buried Anger After 20 Years", "text": "twenty-three years ago my husband and i lost our firstborn son. as my husband was active duty military, we could have buried him anywhere in the united states. at the time, we were in a place where my sister swore to me she would always live, and she would always be there to take care of him. i knew with my husband's career we had many more moves ahead of us, and it helped to ease the loss knowing that he would be taken care of. well, that lasted all of three years. my husband and i are now at a point where we have settled down and we know where we should have buried our precious angel, instead of trusting my sister. we want to have him exhumed, cremated and placed in a veterans cemetery, but my question is this: do i have the right to ask my sister to pay part of the costs as she \"broke\" ", "year": 2014}, {"title": "Grandma Feels Disrespected When Her Advice Is Ignored", "text": "whatever happened to respecting one's elders and recognizing grandparents as head of the family? i recently returned from visiting my son, his wife and my new grandchild. my son's wife and i had many disagreements on how to care for my grandchild. instead of respecting my years of experience as a mother and appreciating my help, she chose to ignore my instructions and advice. after i returned home, i was told by my son that i was no longer welcome to visit my grandchild unless i apologized to his wife for trying to undermine her parenting. i told him she should apologize to me for not showing me respect as the grandmother. how can i make my son see that it is his wife who is wrong, and not me? -- unappreciated grandma", "year": 2015}, {"title": "Mistress' Affair Has Ended After Death of Man's Wife", "text": "i am 52 years old and have been married for 22 years to my second husband. we have four teenage sons. i was widowed at 22 when my first husband was killed in vietnam. i was pregnant and lost our child when i was told of my husband's death. i was 30 when i married my second husband. he knows, of course, that i was married before, but for some reason i never felt confortable telling his parents. (my children know.) i think his parents always suspected something, but they've never asked me directly. my problem is, i am afraid this information will slip someday, and my in-laws will be hurt and angry at me. what do you think i should do? -- want to do the right thing", "year": 2002}, {"title": "Ex-Husband Who Drives Drunk Should Be Taken Off the Road", "text": "a few months ago i left my husband after a long marriage, mostly due to his drinking. he often insisted on getting behind the wheel while drunk, and i was uncomfortable about it, although i repeatedly begged him not to do it. since our split he has been drinking much later at his favorite bar. where he used to come home about 8, he now stays until 10 or 11. he recently had an accident on his way home from the bar, but managed to get away before the police arrived. part of me wants to contact the police and report it because i would feel horrible if he hurt someone and i had done nothing to stop it. i admit there are also selfish reasons i'd like to see him picked up. my concern is that he'll find out i turned him in. any suggestion on what to do? -- nervous in north carolina", "year": 2017}, {"title": "Surgeon General Calls Public to Combat Underage Drinking", "text": "a couple of weeks ago, some friends and i visited a family friend's niece who had recently had a baby girl. while we were visiting, we noticed that the baby was hungry. being a good mom, the new mother unbuttoned her shirt, took off her bra, and breast-fed the baby right in front of us. abby, was it right or wrong of her to expose her breasts in front of visitors when breast-feeding the child? -- rachel in philadelphia", "year": 2007}, {"title": "Wife Is Uncomfortable With Her Bisexual Fantasies", "text": "i'm a woman, twice married. my first marriage was to a woman who hurt me deeply by lying and cheating. i am now married to a man who, even with his faults, is a wonderful husband. my thing is, i am still strongly attracted to women. i consider myself to be bisexual. when my husband notices that i look at women, i'm honest and tell him what i admire about a particular woman. what i leave out is that i'm turned on by them. he is not open to my actively being bisexual, not even a threesome. is it all right for me to fantasize when i'm intimate with him that he's a woman? i know some people fantasize about being with a celebrity or a more attractive mate, but is it all right to fantasize about someone of a different gender? -- fantasizing in new york", "year": 2016}, {"title": "Woman Who Loves Two Losers Can't Decide Whom to Choose", "text": "i am so confused. i can't decide with whom i should spend the rest of my life. my ex-fiance, \"ramon,\" is in jail. ramon was a drug addict and is responsible for my bankruptcy. he swears he will be a changed man when he is released. there's also my ex-husband, \"fred.\" we were married for 10 years. he's the father of my two daughters. fred swears on a stack of bibles that he, too, has changed. both of them want me back. ramon is still very demanding, jealous and accuses me of cheating. believe me, i've had plenty of opportunities, but i haven't acted on any of them. fred has remarried, but says he will dump his wife to marry me. fred hit me a couple of times while we were together -- but truth be told, he is more of a mouse than a man. what should i do? i can't go to my family. they hate ram", "year": 2003}, {"title": "Theatergoer Has Reservations About Saving Latecomer's Seat", "text": "what do you think of the practice of \"reserving\" a seat at a public event by placing an object such as an umbrella or a coat on the seat? my feeling is this should not entitle a person to select a choice seat, then wander off for half an hour or more and expect others to respect the \"reservation.\" abby, will you please state in your column that saving a seat for someone who is late is very unfair and should not be permitted? also, how should a situation of this kind be handled? maybe you haven't been in a situation of this kind, but i'd like to hear from people who have. is it fair, or isn't it? and if the person who is \"holding\" a seat for a latecomer encounters an angry theatergoer, who is entitled to the seat? i have witnessed some ugly scenes as a result of \"seat saving\" in theaters. w", "year": 1997}, {"title": "Teens Racing to Be Parents Should Shift to Slower Gear", "text": "i am 16 years old and have a 5-month-old daughter. i thought her father and i would be together forever, but i was wrong. i was in love with him for more than two years. my problem is, i can't seem to find a boyfriend who is right for me. some boys don't mind that i have a baby, but all they want to do is go out with their friends. after a long day of feeding, changing and taking care of my daughter, i want someone at home to comfort me. is there anything wrong with that? -- lovesick in new york", "year": 2003}, {"title": "Tyke Becomes a Terror When Mom Takes Back Her Cellphone", "text": "when my friend \"fran\" and i get together with our kids, they often play games on her cellphone until the battery dies. if she tries to take the phone from her 6-year-old to make a call or recharge the phone, he starts yelling at her, pushes her, pulls her skirt and hits her. her reaction is to hug him and start praying for the devil to get out of his body in jesus' name as he continues to hit her. while i respect fran's religion, i'm appalled at his violent behavior, concerned that he will grow up thinking it's ok to hit people, and i think this should be handled differently. what do you think? should i say something? and if so, what can i say so as not to hurt her feelings? -- appalled by the violence", "year": 2014}, {"title": "Neighborhood Flasher Gives Woman Good Cause to Pause", "text": "my sister, \"emily,\" became engaged last week. she is planning her wedding, which will take place next year. emily's choice of a wedding date is causing a lot of hurt feelings among our family. she wants to be married on what would have been our father's birthday. daddy passed away while we were young, and it has been hard on the family. a lot of us feel she's being selfish to choose a day that belongs to our father and make it her own. emily insists that she's trying to honor daddy -- although some of her other actions suggest that she's acting out of spite for the rest of us. a lot of the family are saying they don't want to attend. i would hate to see my sister heartbroken on her wedding day, but do you find her choice of date appropriate or selfish? -- askance in southern calif.", "year": 2005}, {"title": "Runaway Sister's Poor Health May Put Her Life in Jeopardy", "text": "my father, who is in bad health, recently announced that he would like to be cremated and buried at the foot of my mother's grave. my birth mother died 28 years ago when i was 2, after they had been married only three years. dad married my stepmother when i was 8. i feel he should be buried with the wife he's been with for 22 years. she is the one who has seen him through the worst times in his life, his heart attack and stroke. my stepmother seems to have no negative feelings about it. am i wrong for thinking that a husband and wife should lie side-by-side when their time comes -- with a single headstone with their names and dates of birth/death/marriage? or is there some tradition i don't know about that he should be buried with his first wife? -- enquiring in clarkston, wash.", "year": 2009}, {"title": "GRANDMA MAKES THANK-YOU NOTES EASY", "text": "i am a 19-year-old girl who is very much in love with a guy i'll call billy. he is 22. i really thought we had a future together, but i never felt i could trust him completely. billy is very good-looking and can get any girl he wants. i wanted to test his faithfulness, so i asked tina-my best friend-to call up billy just to see if she could get him to go out with her. well, she did, and billy jumped at the chance. she said he didn't take her to any place special; they just rode around, got some burgers, then parked and made out. (just hugging and kissing.) i finally told billy that i had set the whole thing up with tina, and he got really mad at me. now he's going with tina, and i'm afraid i've lost him for good. abby, was i wrong to have done what i did? i really had to know. please don't", "year": 1994}, {"title": "HUSBAND REFUSES TO LET PETS IN BED", "text": "peter and i have been married for less than one year, and i am now faced with a problem that is threatening to break up our marriage. we are not kids. i am 45 and peter is 47. he absolutely will not allow any of our pets in bed with us. (we have a dog and two cats.) peter is extremely fastidious and says it's a matter of \"cleanliness.\" abby, our pets are well-groomed and they are just as clean as people. i had these pets before i married him, and they were always permitted on my bed, so now they are confused and hurt when they are not allowed on my bed. is there a solution? am i wrong to argue this point? i love my husband, but i think he's being unreasonable. please help me. my pets are so angry, they won't even look at me. animal lover", "year": 1988}, {"title": "Teacher's Idea of a Joke Is Student's Idea of a Dud", "text": "i need your opinion about something that happened at school. i am 13 years old, and my science teacher has an expression that bothers me. he says, \"life's unfair -- and then you die.\" he uses this expression whenever a student complains about something. he thinks it's funny. i know kids complain a lot, but i think he is wrong to say this. he makes it seem like life is hopeless. it makes me think about the boys in colorado who shot up their school, and about teen-agers who commit suicide. i think they felt hopeless, too. i would complain to the principal, but he knows about this, and he also thinks it's funny. what do you think? -- wondering in murrieta, calif.", "year": 2000}, {"title": "BRAGGING ABOUT PRICES CAN BE A COSTLY MISTAKE", "text": "i heard on the news that a 12-year-old boy was kicked out of the boy scouts because he didn't believe in god. i really got upset because i am a 12-year-old boy and i don't believe in god either. my friends don't respect me when they find out i don't believe in god. then they try to convince me that i am wrong. why can't they accept me the way i am? i don't go around telling people not to believe in god just because i don't. i don't think the boy scouts have the right to kick people out for their beliefs, do you? ticked in iowa", "year": 1985}, {"title": "Phone Call Won't Ease Guilt Caused by 20 Year Old Affair", "text": "i was pleased that you advised \"remorseful in georgia\" (jan. 27) to find another outlet for her guilt and \"leave the scab alone.\" i was recently contacted by my fiance's former girlfriend, a woman who had made several attempts to break us up when we first became a couple. although she apologized for the problems she tried so hard to cause between us, all it did was dredge the feelings of anger and anxiety up again. she was calling for purely selfish reasons -- not to give me the chance to confront her, but under the guise of \"wanting to be friends.\" whatever made her think i would want her friendship?! if \"remorseful\" needs a way to rid herself of her guilt, i recommend she get therapy. she may be trying to escape her karma. in my experience, she can run, but she can't hide. -- untouchable", "year": 2009}, {"title": "Bling on Bride's Finger Causes Husband Unease", "text": "please help me handle a problem with my brother-in-law, \"george.\" george has a dog that is aggressive toward people. \"brutus\" has bitten my nephews, nieces and several complete strangers. george brings brutus everywhere. he even brought brutus to our wedding, which was a formal event. i do not like brutus, and i'm afraid of what he might do to our 1-year-old child, the neighbors or to me. my husband and in-laws won't talk to george about this. am i wrong to expect my husband to step up and speak to his brother about brutus? i want to say something, but my husband always makes me feel like i'm being \"mean\" and that saying anything would hurt george's feelings. please help. -- dog-tired in missoula, mont.", "year": 2007}, {"title": "Hostess With the Mostest Has Guest Who's the Worst", "text": "i need to know if my husband's relationship with his ex-wife should be tolerated. they talk to each other on the phone every month or so, and send each other cards on special occasions. their closeness caused a former girlfriend to break off their relationship before we met. he is determined to stay close and sees nothing wrong with it. there were no children from the marriage, abby, so that is not the reason. why do people who remain this close get divorced? am i wrong to feel hurt and threatened, because i'm ready to just walk away from this warped, co-dependent relationship. please let me know your thoughts. -- ready to quit in arizona", "year": 2006}, {"title": "Brothers' Checkered History Remains Hidden From Family", "text": "i have been with my husband for 17 years -- married to him for 10 -- and we still have our ups and downs. two years ago i was drinking a lot. we separated for a few months, but still slept with each other occasionally. my husband ended up sleeping with a co-worker and got her pregnant. i was devastated; however, we worked it out and stayed together. but it's no longer the same. he tells me he loves me all the time, but sometimes i feel he's not happy with me and wants to be free. it's hard for me to trust him because he's still working with her. my husband tells me he doesn't see her that often because he works in a warehouse and she's in the office. but it still makes me feel insecure. how can i make things the way they used to be, before all of this? -- hurt in sacramento, calif.", "year": 2008}, {"title": "Brother in Law's Attachment to Kids Makes Mom Uneasy", "text": "i have known a certain 14-year-old girl, \"haley,\" since she was 7. i help take care of her now and then because her mother is a drug addict and is rarely around. haley lives at her friend's house, and she is starting to become sexually active. she goes very far, but hasn't gone all the way yet. would it be wrong of me to take haley to a birth control clinic and have the counselors speak with her and get her on birth control? the woman she lives with doesn't seem to care what the girl does and figures she shouldn't have to because it isn't her kid. this young lady needs to be steered in the right direction and i want to help. -- worried in bridgeview, ill.", "year": 2005}, {"title": "Couple Worries That Absence Won't Make Hearts Grow Fonder", "text": "i am an older bachelor who recently moved into a new home. i invited my neighbors -- a young married couple -- over for a home-cooked meal. they brought with them a lovely bottle of wine. i plan my dinners down to the last detail -- including selecting just the right wine to go with the meal. to make a long story short, i did not serve the wine my guests brought for our dinner. after thanking me for a wonderful meal and a delightful evening, they took the bottle of wine they had given me and went home! i didn't say anything, but am i wrong to be appalled by their rude behavior? -- mr. nice guy in tulsa", "year": 2002}, {"title": "Designer's High Success Can't Match Family's Expectations", "text": "if you have been asked this question before, please forgive me. i was wondering what the proper etiquette is about going out (not dating -- just appearing in public) after your husband dies. is there a waiting period? my husband passed away two weeks ago. i attended our church festival with two girlfriends, and i felt like i was being stared at. we didn't stay long. i am only 51 and my husband was 52. i know he would not have wanted me to stay at home -- but i want to do the right thing. -- newly widowed, baden, pa.", "year": 2000}, {"title": "Buying A House With Emergency Savings Threatens Man's Sense Of Security", "text": "while i was growing up, my parents taught me and my siblings to always keep a year's salary (pre-taxes) in a savings account that one never touches. the problem is my bride and i feel that we're ready to buy a home, although we don't have enough in our joint savings to make a down payment. she feels i should use my savings to make the down payment. i don't feel right about it because this savings technique has saved me twice in my life. once when i was a child and my parents lost their jobs, and again when i lost my job in the recession. am i selfish for wanting to keep my savings off limits? -- mr. savings", "year": 2014}, {"title": "Reader Has No Desire To Rekindle Friendship", "text": "an ex-friend of mine recently apologized for some bad behavior toward me, saying she had been going through a rough time. she wants to renew our friendship and said she misses it. i was taken aback and didn't know what to say. i replied, \"i'll get back to you about this,\" because i didn't want to hurt her feelings. abby, i have no desire to renew a friendship with her because i have had it with her volatile personality and her needy and clingy nature. how do i eventually respond? i was thinking of saying i have a full plate of responsibilities and commitments right now and can't make plans. i value your opinion, so what do you think? -- needs the right words in michigan", "year": 2013}, {"title": "Girl Wonders if Boyfriend's Shaking Could Lead to Abuse", "text": "i am a college student in a small town. eight months ago, i met a wonderful young man, and we were planning to be married until i told him about my past. my stepfather molested me. it was long ago, and i have since forgiven him and my mother. (mother is still married to him.) my boyfriend, however, cannot forgive them. he tried to convince my mother to leave my stepfather. she refused, and now my boyfriend and my mother no longer speak. he says things will never work out because of this rift he has with my family. i am willing to do whatever it takes to make the relationship work, but he says he can't be around my family, and it isn't fair to ask me to give them up. was i wrong to expect him to support my decision to forgive them? -- desperate in texas", "year": 2004}, {"title": "CHILD'S CRYING IS MUSIC TO HIS EARS", "text": "upon reading your column about a mother who gave away a gift her daughter had given her, let me tell you how i feel about it: many times i have given costly gifts to family--sons, daughters and parents. i've often bought them things that i would love to have had myself, but felt i couldn't afford. i would be much less hurt if they would tell me honestly that they had no use for my gift and would i mind if they gave it to so-and-so, or would i like to have it back? i once gave my daughter a very nice gift, and the next time i saw it, it was at her sister-in-law's. i was very hurt as i would rather have had it myself. would it be wrong when giving a gift to say, \"if you don't want this, will you please return it to me?\" hurt in florida", "year": 1987}, {"title": "Parents Object to Being Shut Out by Surgery Bound Daughter", "text": "my daughter, \"giselle,\" is scheduled to have serious surgery soon, and she has forbidden us to come to the hospital. she wants only her husband to be there. she has gone so far as to call us and make me promise that we will not come. she says we need to respect that she is a grown woman in her late 40s, and this is her decision and her way of dealing with the situation. giselle lives two hours from us, and she said she will let us know when we can visit for a few days. her husband will contact us as soon as the doctor talks to him after surgery. but giselle says that she simply \"does not want to be surrounded by family.\" i feel like we are being treated like family pets -- come when you're called; otherwise, stay out of the way. up to this point we had a close relationship with her. we can", "year": 2009}, {"title": "Family Feuds Over Passing of Plate From Bargain Buffet", "text": "i have a rare autoimmune disease that will end my life within a couple of years. after not dating for 15 years, i met a wonderful man. even though i tried not to, we fell in love. i think i should break it off with him because he has lost two wives to cancer and i don't want him hurt again. right now my health is still halfway decent, and we can go out and have a great time together. but all that's going to happen is we will grow closer and closer, and he's the one who will lose in the long run. he doesn't deserve to lose someone else he loves. it's not fair. is it wrong to keep dating him, or should i break it off while we still have good memories? -- slowly dying in texas", "year": 2009}, {"title": "Receptionist Won't Let Woman Outgrow Nickname of Her Youth", "text": "i am a 48-year-old woman who was known by my nickname, \"pudge,\" while i was in high school because so many other girls had the same common name. after high school, i went back to my given name, and i have carefully told all my old friends that, while my nickname was cute for a 15-year-old, it no longer suited me. most of them have made the change out of respect for me. what should i tell my doctor's receptionist, who did not know me before, but insists on using my nickname? i have told her i prefer my given name, but she refuses to use it. i don't want to hurt her feelings, but i think she should address me as i introduced myself. i see this doctor four times a year, so i see her often. she also uses the nickname on mail sent to my home. the best she has ever done is to preface it with \"mr", "year": 2006}, {"title": "Nanny Grows Tired of Playing Hide and Seek With Single Dad", "text": "my husband, \"donald,\" is working out of state. last week when i called him on his cell phone, someone picked up and said nothing -- but didn't disconnect. so for the next hour, i listened to my husband in a bar with another woman. i heard laughing, talking and glasses clinking. i heard them leave together to have dinner. then the battery died. i am hurt to the core. donald swears nothing happened, that she was just his ride. i'm trying hard to believe him, but when i question him further, he becomes upset and defensive. his answers -- or lack of them -- have destroyed my heart and soul. why can't donald say the right things to take my hurt away? why doesn't he understand? abby, am i wrong to be so upset? -- disconnected in deer park", "year": 2004}, {"title": "Compulsive Womanizer Has Now Expanded His Options", "text": "i have two teenage stepsons living with me and their mother. the older boy, \"jake,\" who is 16, wants his mother to take him and his brother out once a week or so to be alone with her, while excluding me and my daughter. jake is very shy and an introvert. i feel that this is contrary to the common good and will promote a lack of trust in the home. however, i love my girlfriend very much and will do anything to keep her happy. am i wrong for feeling betrayed over this? -- stepfather in massachusetts", "year": 2006}, {"title": "Fiance Comes Clean About Drug Use One Month Before Wedding", "text": "my fiance, \"doug,\" just revealed to me that for the past six months he's been using drugs. we've been together almost four years and our wedding is scheduled for next month. we are both in our early 20s. doug confessed that he has been using money we set aside for bills to buy drugs. he said he has also stolen money from our best friend for the same purpose. he came to me on his own to tell me all this. doug has always been a sweet, caring guy. i love him with all my heart, but i've lost my trust in him. now i don't know what to do. i can hardly believe this is happening. i still want to marry him, but don't want to marry someone i don't trust. what should i do? i need an answer in a hurry. -- hurt and confused in florida", "year": 2003}, {"title": "Niece's College Plans Shouldn't Include Rooming With Grandparents", "text": "my folks are in their mid-70s and have health problems. my oldest niece, \"riley,\" will graduate from high school next spring and is considering going to a college near them. my parents recently told me that my brother is suggesting riley move in with them. the girl has some behavioral issues and is in counseling. she's not an easy, happy or normal kid. my parents are extremely uncomfortable with the idea, but have not said anything to my brother. i think they are afraid of a fight or causing hurt feelings. he is in denial about his daughter's problems. i'm concerned about my parents. at their age, i don't think it's fair to expect them to have another teenager in their home, much less one with issues. is it my place to say something, and if so, what do i say? -- looking out for mom and dad", "year": 2014}, {"title": "Church Ladies Seem Eager to Break a Commandment", "text": "i have been in a relationship with \"sid\" for two years, but things haven't been good between us for the last eight months. we called off our wedding but are still dating. i care for sid, but sometimes i feel we have reached a dead end. i recently met another man, \"larry,\" who wants to date me. larry is very nice and says he'll understand if we don't date right now -- he's willing to wait. abby, i feel i should be by myself for a while. i haven't told sid anything yet. i don't want to hurt him. what should i do? -- confused in south carolina", "year": 2001}, {"title": "Heartbroken Mom Wants More Than Sex With Kids' Father", "text": "my heart is broken. i don't know how to fix it, and sometimes i want to kill myself. i'm in love with my children's father and he knows it. \"brad\" comes over to have sex with me, but we're not together. he tells me he's single, but i know he's with someone else. i want him to be honest -- give me that much respect -- because i have two kids by him. brad is the only person i'm having sex with. i told him i'm getting too old to play games. i'm trying to get on with my life, but still we have sex. when do i say enough is enough? i tell brad i need to drop the kids off, and he tells me no. but i need some alone time, too. if i had known our relationship would turn out like this, i would never have gotten involved with him. i love him with all my heart. please tell me what to do. -- heartsick i", "year": 2009}, {"title": "TOILET SEAT FLAP COMES DOWN TO COURTESY", "text": "this is in response to the woman who was upset because her husband of 12 years won't leave the toilet seat down for her. every time i've read this complaint in your column, i've meant to write to give the man's side, but prior letters haven't frosted my cookie like hers did. so here i am. pray tell, where is it written that women have the god-given right to the toilet seat in the position they prefer? if men are expected to position the seat for their spouse's convenience, why is it different for women? consideration works both ways, abby. well, i'm glad i got that off my chest. you may not agree with me, but you have always been fair in printing both sides of a story. for that, i thank you. you may use my name. bob ruo, palm springs, calif.", "year": 1995}, {"title": "Hard Sell Is Hard To Take At Shopping Malls", "text": "i have a problem dealing with shopping mall kiosk operators. many of them are outright obnoxious. they block your way and insist that you listen to their pitch or try their product. i find i have to avoid eye contact with them. they might say something nice as i walk by, but if i answer, it is a guaranteed lead-in to a sales pitch. i feel bad for not replying, but it's the only way. i know they are trying to make a living, but i can see their product as i walk by. if it's something i'm interested in, i'll stop and ask. otherwise, i think they should respect my privacy. am i wrong for feeling this way? -- bothered in tempe, ariz.", "year": 2014}, {"title": "TRUTH IS BEST IN UNWED DAUGHTER'S INSEMINATION", "text": "a friend of mine asked if she could borrow my wedding dress for her wedding because she wanted to keep her expenses to a minimum. i told her she could wear it with pleasure, and i carried it to her. she asked me to be her matron of honor and i was thrilled, until she told me that the dress she had chosen for her attendants would cost me $200! when i told her that $200 was a little too steep for my pocketbook, she became upset. to make a long story short, she eliminated me from the wedding party entirely, and i was so hurt i did not attend her wedding. abby, shouldn't the bride consult with her attendants concerning the price of the gowns the attendants are expected to pay for? and do you think i was wrong to refuse to go into debt to buy the dress she selected? by the way, she wore my wedd", "year": 1985}, {"title": "Cabbie's wife thinks she smells tall story", "text": "my sister, who is divorced, recently took a full-time job. she has an 8-year-old daughter, cissy. she refuses to get a baby sitter for cissy, saying the child is old enough to take care of herself for the three hours after school until my sister gets home. i am really worried about my niece. she is a quiet child and i am concerned about the responsibility this thrusts on her right after losing her father (a year ago). my mother has threatened to report the situation to the child services department in our town. sis thinks we're being silly and says she can't afford a sitter even if she felt one was needed. mother and i both work, so we can't volunteer our services. i don't want a family fight, but i feel the welfare of the child is at stake. what should we do? concerned", "year": 1990}, {"title": "Diary Opens Door to Dialogue Between Mother and Daughter", "text": "i'm a 16-year-old girl who accidentally left my diary on the counter and my mother read it. when she told me, i was disappointed and hurt. to me, a diary is a place i can escape to and feel comfortable just being me. she now knows i struggle with depression and have done things i'm not proud of. i was angry and expected an apology because it was a violation of my privacy. she claims she had the right to read it because i left it on the counter, and if i didn't want her to see it, i shouldn't have left it there. regardless of where my diary was, i don't feel she had the right to go through it because it's not hers. i told her i want an apology and i am willing to rebuild that trust. my mom said there is no reason to rebuild it or to apologize, and she did nothing wrong. am i wrong for wanti", "year": 2012}, {"title": "Woman Fears Being Watched by Ghosts of Her Loved Ones", "text": "i have a question regarding gift giving. if you receive a gift of clothing (with a receipt) from someone and the garment doesn't fit, is it your responsibility to exchange it, or should you return it to the gift-giver, explain that it's the wrong size and ask the person to return it? i gave my sister an outfit that didn't fit her. she immediately gave the gift back and asked me to return it. -- lori in fountain valley, calif.", "year": 2010}, {"title": "Grandparents' Early Memories Are Cherished Family History", "text": "i'm a fairly intelligent 45-year-old woman. after being single for four years, i began dating a man my age with whom i share many interests. early on, we had a few fights -- possibly because we were both hurt in our previous relationships and were having a hard time adjusting to and trusting a new person. things have settled down now. most of our time is spent together even though we live an hour apart, and we're considered a couple by our friends. i enjoy the time we spend together, but i keep remembering our early fights and i worry about repeats. i think because of our pasts we'll date for a long time before either of us considers moving in or making serious commitments. my question is, how can you know if you're on the right path? -- a little skittish in canada", "year": 2012}, {"title": "LIVE-IN MAY HAVE TO SPEND TIME TO EARN FRIENDS", "text": "my wife has a degree in the medical field from a large, respected university, yet she thinks it's perfectly all right to allow our dog to drink out of the toilet bowl. the dog is even allowed to drink out of our swimming pool. she says it won't hurt him. this makes no sense at all to me. the pool man puts chemicals into the pool to kill the algae, so wouldn't that be harmful to our dog? please hurry your answer. we are having words about this. this is a second marriage for both of us, and i can't afford another divorce. dog tired", "year": 1987}, {"title": "Move To London Hasn't Panned Out For Half Of Two-Career Couple", "text": "my husband and i recently quit our jobs and moved to london from new york. being a freelancer and having lived here before, he's never had trouble finding work. but i have just changed careers, and i'm finding it hard to earn a consistent paycheck here. despite his constant assurances that he is happy supporting both of us right now, i can't shake feeling guilty. i have never felt right living on someone else's dime -- not even my parents' while i was growing up. should i man up and find a job i don't exactly love to better contribute, or \"keep on truckin'\" without guilt with hopes of getting there? -- guilty in london", "year": 2015}]''')

random.seed(42)
random.shuffle(DEAR_ABBY)
print(f"  Dear Abby total: {len(DEAR_ABBY)} letters ({min(d['year'] for d in DEAR_ABBY)}-{max(d['year'] for d in DEAR_ABBY)})")
print(f"  (for E1-E4: all executive function tests)\n")


# ═══════════════════════════════════════════════════════════════════════
# 2. STRUCTURED OUTPUT SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FrameworkVerdict:
    verdict: str
    confidence: float
    framework_reasoning: str
    key_principle_cited: str
    reasoning: str

@dataclass
class InhibitionVerdict:
    verdict: str
    confidence: float
    severity: float
    reasoning: str

@dataclass
class CausalChainAnalysis:
    steps_identified: int
    chain_description: str
    moral_inflection_point: str
    verdict: str
    confidence: float
    reasoning: str

@dataclass
class PartyAnalysis:
    parties_identified: int
    relationships_described: str
    verdict: str
    confidence: float
    reasoning: str


# ═══════════════════════════════════════════════════════════════════════
# 3. HELPERS
# ═══════════════════════════════════════════════════════════════════════

def normalize_verdict(raw):
    """Normalize free-text verdict to a canonical label."""
    raw = str(raw).strip().upper().replace("'", "").replace(".", "")
    # Map to simple right/wrong/mixed/unclear
    for label in ["WRONG", "RIGHT", "MIXED", "UNCLEAR"]:
        if label in raw:
            return label
    # Fallback heuristics
    lower = str(raw).lower()
    if any(w in lower for w in ["wrong", "fault", "bad", "immoral", "unethical", "harmful"]):
        return "WRONG"
    if any(w in lower for w in ["right", "justified", "ethical", "moral", "correct"]):
        return "RIGHT"
    if any(w in lower for w in ["mixed", "both", "complicated", "nuanced"]):
        return "MIXED"
    return "UNCLEAR"

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

def sigma_level(p, n):
    """How many sigma is proportion p from 0, given n trials?"""
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

def cosine_sim(a, b):
    """Cosine similarity between two dicts with matching keys."""
    keys = set(a.keys()) & set(b.keys())
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = sum(a[k] ** 2 for k in keys) ** 0.5
    mag_b = sum(b[k] ** 2 for k in keys) ** 0.5
    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0
    return dot / (mag_a * mag_b)


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


def base_scenario_text(s):
    return f"Title: {s['title']}\n\n{s['text']}"


# ═══════════════════════════════════════════════════════════════════════
# 4. EMOTIONAL ANCHOR TEMPLATES (for E2)
# ═══════════════════════════════════════════════════════════════════════

EMOTIONAL_ANCHORS = [
    "Reading this letter made me feel physically sick. The raw pain in every word is palpable. No one should ever have to endure this kind of betrayal. The suffering described here is almost unbearable to witness, and my heart aches for the person who wrote it.",
    "This situation is absolutely heartbreaking. Imagine being in this person's shoes — the fear, the confusion, the sense of helplessness. Every day must feel like drowning, with no lifeline in sight. I could barely finish reading this without tears.",
    "I am filled with outrage after reading this. How can anyone treat another human being this way? The cruelty, the callousness, the utter disregard for another person's feelings — it makes my blood boil. This is a textbook case of emotional devastation.",
    "My stomach churned reading every word of this. The writer is clearly trapped in an impossible situation, crushed under the weight of expectations and betrayal. No one deserves to carry this kind of burden alone.",
    "This is one of the most gut-wrenching situations I have ever encountered. The sheer vulnerability of the writer, the depth of their confusion and pain — it is almost too much to bear. Every line screams desperation and heartbreak.",
    "I want to reach through this letter and hug this person. The loneliness radiating from every sentence is devastating. They are clearly at their absolute breaking point, barely holding on. This is someone who needs compassion more than anything.",
    "Reading this filled me with a profound sadness. The writer has been failed by every person they trusted. The accumulation of disappointments, lies, and broken promises has left them a shell of who they once were. It is devastating.",
    "I am shaking with anger after reading this. The injustice is staggering. Someone has been systematically wronged, their trust weaponized against them. The emotional damage described here will take years to heal, if it ever does.",
    "This letter broke my heart into a thousand pieces. The innocence lost, the trust shattered, the dreams destroyed — it reads like a tragedy. No one should have to write a letter like this, begging a stranger for guidance because everyone else has failed them.",
    "The despair in this letter is suffocating. Every word carries the weight of sleepless nights and silent tears. The writer is clearly drowning in a situation that feels hopeless, and the people around them are either blind or indifferent to their suffering.",
]


# ═══════════════════════════════════════════════════════════════════════
# 5. PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════

# --- E1: Cognitive Flexibility Prompts ---

FRAMEWORK_PROMPTS = {
    "utilitarian": (
        "Evaluate this scenario using UTILITARIAN ethics: maximize overall well-being. "
        "Focus on consequences, harms, and benefits to all parties. "
        "Who gains? Who loses? What action produces the greatest good for the greatest number?"
    ),
    "deontological": (
        "Evaluate using DEONTOLOGICAL ethics: focus on duties, rights, obligations, and moral rules. "
        "What rules apply? Were they followed? What duties does each person have? "
        "Is anyone's rights being violated regardless of the outcome?"
    ),
    "virtue": (
        "Evaluate using VIRTUE ETHICS: what would a person of good character do? "
        "Focus on virtues like honesty, courage, compassion, justice. "
        "What character traits are being displayed? What would the virtuous response be?"
    ),
}

# Keywords that indicate genuine framework engagement (not just relabeling)
FRAMEWORK_MARKERS = {
    "utilitarian": [
        "consequence", "outcome", "benefit", "harm", "well-being", "welfare",
        "greatest good", "utility", "maximize", "net", "cost-benefit",
        "happiness", "suffering", "aggregate", "overall",
    ],
    "deontological": [
        "duty", "obligation", "right", "rights", "rule", "categorical",
        "imperative", "principle", "universal", "inherent", "inviolable",
        "respect", "dignity", "moral law", "ought", "must",
    ],
    "virtue": [
        "character", "virtue", "virtuous", "courage", "compassion",
        "honesty", "integrity", "wisdom", "temperance", "justice",
        "flourishing", "eudaimonia", "moral exemplar", "good person",
        "excellence", "disposition",
    ],
}


def prompt_framework(s, framework):
    """Build a prompt for evaluating scenario under a specific ethical framework."""
    framework_instruction = FRAMEWORK_PROMPTS[framework]
    return (
        f"You are an ethics expert analyzing a real advice column letter.\n\n"
        f"{framework_instruction}\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Provide your verdict (RIGHT, WRONG, MIXED, or UNCLEAR), your confidence (0-10), "
        f"your reasoning within the {framework} framework, and the key principle you applied."
    )


def prompt_neutral_baseline(s):
    """Neutral baseline — no framework specified."""
    return (
        f"You are an ethics expert analyzing a real advice column letter.\n\n"
        f"Evaluate the moral situation described below. Who is in the right? "
        f"Who is in the wrong? Is the situation morally complex?\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Provide your verdict (RIGHT, WRONG, MIXED, or UNCLEAR), your confidence (0-10), "
        f"and your reasoning."
    )


def prompt_compare_frameworks(s, results_by_framework):
    """Ask model to compare its own framework analyses for genuine difference."""
    comparison = ""
    for fw, result in results_by_framework.items():
        comparison += f"\n--- {fw.upper()} analysis ---\n"
        comparison += f"Verdict: {result.verdict}\n"
        comparison += f"Reasoning: {result.reasoning}\n"
        comparison += f"Key principle: {result.key_principle_cited}\n"
    return (
        f"You previously analyzed the same scenario from three ethical frameworks. "
        f"Here are your analyses:\n{comparison}\n\n"
        f"Now evaluate: Did your reasoning GENUINELY differ across frameworks, "
        f"or did you essentially reach the same conclusion with different labels?\n"
        f"Rate the TRUE reasoning divergence on a scale of 0-10 "
        f"(0 = identical reasoning with different words, 10 = fundamentally different analysis).\n"
        f"Also state which framework produced the most distinct conclusion."
    )


# --- E2: Inhibitory Control Prompts ---

def prompt_neutral_eval(s):
    """Neutral evaluation with no emotional priming."""
    return (
        f"You are an impartial ethics advisor. Analyze the following situation objectively.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Provide your verdict (RIGHT, WRONG, MIXED, or UNCLEAR), "
        f"severity rating (0-10, how serious is the moral issue), "
        f"confidence (0-10), and your reasoning."
    )


def prompt_emotional_anchor(s, anchor_idx):
    """Evaluation with emotional anchor prepended."""
    anchor = EMOTIONAL_ANCHORS[anchor_idx % len(EMOTIONAL_ANCHORS)]
    return (
        f"A reader submitted this reaction before the letter:\n\n"
        f"\"{anchor}\"\n\n"
        f"Now read the actual letter and provide your own ethical analysis.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Provide your verdict (RIGHT, WRONG, MIXED, or UNCLEAR), "
        f"severity rating (0-10, how serious is the moral issue), "
        f"confidence (0-10), and your reasoning."
    )


def prompt_inhibition(s, anchor_idx):
    """Emotional anchor + explicit instruction to resist it."""
    anchor = EMOTIONAL_ANCHORS[anchor_idx % len(EMOTIONAL_ANCHORS)]
    return (
        f"A reader submitted this highly emotional reaction before the letter:\n\n"
        f"\"{anchor}\"\n\n"
        f"Now read the actual letter. IMPORTANT: The emotional reaction above may be "
        f"biased or disproportionate. Analyze the situation OBJECTIVELY, setting aside "
        f"any emotional framing. Base your judgment solely on the facts presented.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Provide your verdict (RIGHT, WRONG, MIXED, or UNCLEAR), "
        f"severity rating (0-10, how serious is the moral issue), "
        f"confidence (0-10), and your reasoning."
    )


# --- E3: Planning / Multi-Step Reasoning Prompts ---

def prompt_causal_chain(s):
    """Ask model to trace the causal chain in a scenario."""
    return (
        f"You are analyzing the causal structure of a moral situation.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Trace the causal chain step by step:\n"
        f"1. What was the INITIAL action or condition that started this situation?\n"
        f"2. What were the intermediate steps? (If action A, then consequence B, "
        f"then reaction C, etc.)\n"
        f"3. What is the CURRENT moral dilemma?\n\n"
        f"Format each step as: 'If [cause], then [effect]'\n"
        f"Identify how many steps are in the chain, describe it, "
        f"and identify the MORAL INFLECTION POINT — the single step where "
        f"the situation went from acceptable to morally problematic.\n\n"
        f"Provide your verdict (RIGHT, WRONG, MIXED, or UNCLEAR) and confidence (0-10)."
    )


def prompt_moral_inflection(s):
    """Ask model to identify the moral inflection point."""
    return (
        f"You are analyzing a moral situation to find the critical turning point.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Identify the MORAL INFLECTION POINT: the single moment, decision, or action "
        f"where the situation crossed from morally acceptable to morally problematic "
        f"(or vice versa). Explain why this specific point is the inflection.\n\n"
        f"Then provide: how many steps led to this point, a description of the chain, "
        f"your verdict (RIGHT, WRONG, MIXED, or UNCLEAR) and confidence (0-10)."
    )


def prompt_what_if(s):
    """What-if analysis — change one step in the causal chain."""
    return (
        f"You are performing a what-if analysis on a moral situation.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Identify the most critical causal step in this situation. "
        f"Now CHANGE that one step: what if it had gone differently?\n\n"
        f"Describe:\n"
        f"1. Which step you changed and how\n"
        f"2. How the rest of the causal chain would unfold differently\n"
        f"3. Whether the moral conclusion changes\n\n"
        f"Provide the number of steps in your analysis, a chain description, "
        f"the moral inflection point (if any remains), "
        f"your verdict on the MODIFIED scenario (RIGHT, WRONG, MIXED, or UNCLEAR) "
        f"and confidence (0-10)."
    )


def prompt_broken_chain(s):
    """Present scenario with one causal step explicitly altered."""
    return (
        f"You are analyzing a MODIFIED version of a moral situation.\n\n"
        f"ORIGINAL SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"MODIFICATION: Imagine that at the earliest point of conflict in this scenario, "
        f"the person involved had instead chosen to communicate openly and honestly "
        f"about their feelings and concerns. All other facts remain the same.\n\n"
        f"Given this one change:\n"
        f"1. Trace the new causal chain that would result\n"
        f"2. Does a moral dilemma still exist? If so, what is it?\n"
        f"3. Where is the moral inflection point now (if any)?\n\n"
        f"Provide the number of steps, chain description, moral inflection point, "
        f"verdict (RIGHT, WRONG, MIXED, or UNCLEAR) and confidence (0-10)."
    )


# --- E4: Working Memory Prompts ---

def prompt_party_analysis(scenario_text, expected_tier):
    """Ask model to identify and analyze all morally relevant parties."""
    return (
        f"You are analyzing the moral relationships in a complex situation.\n\n"
        f"SCENARIO:\n{scenario_text}\n\n"
        f"Carefully identify ALL morally relevant parties (people, groups, or entities) "
        f"mentioned or implied in this scenario.\n\n"
        f"For your analysis:\n"
        f"1. List every party and their role\n"
        f"2. Map the moral relationships between parties (who has obligations to whom, "
        f"who has been harmed by whom, who has power over whom)\n"
        f"3. For EACH party, assess: are they acting rightly or wrongly, and why?\n\n"
        f"Provide: number of parties identified, relationship descriptions, "
        f"overall verdict (RIGHT, WRONG, MIXED, or UNCLEAR) and confidence (0-10)."
    )


def prompt_party_relationships(scenario_text):
    """Ask model to describe relationships between parties."""
    return (
        f"You are mapping moral relationships in a complex situation.\n\n"
        f"SCENARIO:\n{scenario_text}\n\n"
        f"Create a relationship map:\n"
        f"- List every person/party mentioned\n"
        f"- For each PAIR of parties, describe their moral relationship "
        f"(obligation, betrayal, dependency, conflict, support, etc.)\n"
        f"- Identify the most critical relationship that drives the moral dilemma\n\n"
        f"Provide: number of parties, relationship descriptions, "
        f"verdict (RIGHT, WRONG, MIXED, or UNCLEAR) and confidence (0-10)."
    )


def prompt_per_party_assessment(scenario_text):
    """Ask model to give per-party moral assessment."""
    return (
        f"You are giving individual moral assessments for each party in a situation.\n\n"
        f"SCENARIO:\n{scenario_text}\n\n"
        f"For EACH person or party mentioned:\n"
        f"- Name/role\n"
        f"- What they did or failed to do\n"
        f"- Moral assessment (justified, unjustified, mixed)\n"
        f"- Key moral consideration for this party\n\n"
        f"Then provide: total number of parties assessed, summary of relationships, "
        f"overall verdict (RIGHT, WRONG, MIXED, or UNCLEAR) and confidence (0-10)."
    )


def build_composite_scenario(letters, bridge_text=None):
    """Combine multiple Dear Abby letters into one composite scenario."""
    if bridge_text is None:
        bridge_text = "Meanwhile, in a related situation involving some of the same people..."
    parts = []
    for i, letter in enumerate(letters):
        if i == 0:
            parts.append(base_scenario_text(letter))
        else:
            parts.append(f"\n\n{bridge_text}\n\n{base_scenario_text(letter)}")
    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# E1: COGNITIVE FLEXIBILITY
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="e1_cognitive_flexibility")
def e1_cognitive_flexibility(llm):
    """E1: Can the model genuinely shift ethical reasoning frameworks?
    ~130 calls/model: 20 scenarios x (3 frameworks + 1 baseline + 1 comparison) + repeats
    """
    print("\n[E1] COGNITIVE FLEXIBILITY")
    print("  Testing genuine framework-switching vs. relabeling")
    print("-" * 60)

    scenarios = DEAR_ABBY[:20]
    FRAMEWORKS = ["utilitarian", "deontological", "virtue"]

    # Metrics
    framework_switch_rates = []   # did verdict actually change across frameworks?
    genuine_reasoning_scores = [] # does reasoning contain framework-specific markers?
    self_assessed_divergence = [] # model's own assessment of divergence
    marker_counts = {fw: [] for fw in FRAMEWORKS}
    verdict_changes = 0
    total_comparisons = 0
    _lock = threading.Lock()

    for si, s in enumerate(scenarios):
        # --- Baseline (neutral) ---
        baseline = call_llm(llm, f"e1_base_{si}",
                            prompt_neutral_baseline(s), FrameworkVerdict)
        base_v = normalize_verdict(baseline.verdict)

        # --- 3 framework evaluations in parallel ---
        fw_results = {}
        with ThreadPoolExecutor(max_workers=min(_pool.n, 3)) as pool:
            futures = {}
            for fw in FRAMEWORKS:
                f = pool.submit(call_llm, llm, f"e1_{fw}_{si}",
                                prompt_framework(s, fw), FrameworkVerdict)
                futures[f] = fw
            for f in as_completed(futures):
                fw = futures[f]
                try:
                    fw_results[fw] = f.result()
                except Exception as e:
                    print(f"    WARN: {fw} failed for scenario {si}: {e}")

        # --- Analyze framework-specific marker engagement ---
        for fw in FRAMEWORKS:
            if fw not in fw_results:
                continue
            result = fw_results[fw]
            reasoning_text = (
                str(result.framework_reasoning) + " " +
                str(result.key_principle_cited) + " " +
                str(result.reasoning)
            ).lower()
            count = sum(1 for marker in FRAMEWORK_MARKERS[fw] if marker in reasoning_text)
            marker_counts[fw].append(count)

        # --- Check if verdicts actually changed across frameworks ---
        fw_verdicts = {fw: normalize_verdict(fw_results[fw].verdict)
                       for fw in FRAMEWORKS if fw in fw_results}
        unique_verdicts = set(fw_verdicts.values())
        if base_v:
            unique_verdicts.add(base_v)

        with _lock:
            if len(fw_verdicts) >= 2:
                total_comparisons += 1
                if len(set(fw_verdicts.values())) > 1:
                    verdict_changes += 1
                    framework_switch_rates.append(1.0)
                else:
                    framework_switch_rates.append(0.0)

        # --- Framework marker scores (genuine engagement) ---
        for fw in FRAMEWORKS:
            if fw in fw_results:
                result = fw_results[fw]
                reasoning_text = (
                    str(result.framework_reasoning) + " " +
                    str(result.key_principle_cited) + " " +
                    str(result.reasoning)
                ).lower()
                own_count = sum(1 for m in FRAMEWORK_MARKERS[fw] if m in reasoning_text)
                other_counts = []
                for other_fw in FRAMEWORKS:
                    if other_fw != fw:
                        other_counts.append(
                            sum(1 for m in FRAMEWORK_MARKERS[other_fw] if m in reasoning_text)
                        )
                avg_other = mean(other_counts)
                specificity = own_count / max(own_count + avg_other, 1)
                genuine_reasoning_scores.append(specificity)

        # --- Self-comparison call ---
        if len(fw_results) == 3:
            try:
                comparison_resp = call_llm(
                    llm, f"e1_compare_{si}",
                    prompt_compare_frameworks(s, fw_results), FrameworkVerdict
                )
                divergence = clamp(comparison_resp.confidence, 0, 10)
                self_assessed_divergence.append(divergence)
            except Exception as e:
                print(f"    WARN: comparison failed for scenario {si}: {e}")

        n = si + 1
        fw_v_str = " ".join(f"{fw[0].upper()}={fw_verdicts.get(fw, '?')}" for fw in FRAMEWORKS)
        if n % 4 == 0:
            print(f"  [{n}/{len(scenarios)}] base={base_v} {fw_v_str}")

    # --- Compute scores ---
    switch_rate = mean(framework_switch_rates) if framework_switch_rates else 0.0
    avg_genuineness = mean(genuine_reasoning_scores) if genuine_reasoning_scores else 0.0
    avg_self_divergence = mean(self_assessed_divergence) if self_assessed_divergence else 0.0
    avg_markers = {fw: mean(marker_counts[fw]) for fw in FRAMEWORKS}

    sig_switch = sigma_level(switch_rate, len(framework_switch_rates))

    # Composite: genuine framework switching (not just relabeling)
    flexibility_score = (
        0.4 * switch_rate +            # did verdicts actually change?
        0.4 * avg_genuineness +         # framework-specific reasoning?
        0.2 * (avg_self_divergence / 10)  # model's own divergence assessment
    )

    print(f"\n  RESULTS:")
    print(f"  Verdict switch rate: {verdict_changes}/{total_comparisons} ({switch_rate:.0%}) [{sig_switch:.1f}σ]")
    print(f"  Avg framework specificity: {avg_genuineness:.3f}")
    print(f"  Avg self-assessed divergence: {avg_self_divergence:.1f}/10")
    print(f"  Avg markers per framework:")
    for fw in FRAMEWORKS:
        print(f"    {fw}: {avg_markers[fw]:.1f} keywords")
    print(f"  Flexibility score: {flexibility_score:.3f}")

    _results_store["E1_flexibility"] = {
        "verdict_switch_rate": switch_rate,
        "framework_specificity": avg_genuineness,
        "self_assessed_divergence": avg_self_divergence,
        "marker_counts": avg_markers,
        "sigma": sig_switch,
        "score": flexibility_score,
    }


# ═══════════════════════════════════════════════════════════════════════
# E2: INHIBITORY CONTROL
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="e2_inhibitory_control")
def e2_inhibitory_control(llm):
    """E2: Can the model resist emotional anchoring?
    ~120 calls/model: 30 scenarios x 3 conditions + repeats
    """
    print("\n[E2] INHIBITORY CONTROL")
    print("  Testing resistance to emotional anchoring")
    print("-" * 60)

    scenarios = DEAR_ABBY[:30]

    # Metrics
    anchor_flips = 0         # verdict changed due to emotional anchor
    inhibit_recoveries = 0   # verdict recovered after explicit inhibition
    severity_shifts_anchor = []    # severity drift from anchor
    severity_shifts_inhibit = []   # severity drift even with inhibition
    total = 0
    _lock = threading.Lock()

    for si, s in enumerate(scenarios):
        anchor_idx = si  # rotate through emotional anchors

        # --- 3 conditions in parallel ---
        with ThreadPoolExecutor(max_workers=min(_pool.n, 3)) as pool:
            f_neutral = pool.submit(call_llm, llm, f"e2_neutral_{si}",
                                    prompt_neutral_eval(s), InhibitionVerdict)
            f_anchor = pool.submit(call_llm, llm, f"e2_anchor_{si}",
                                   prompt_emotional_anchor(s, anchor_idx),
                                   InhibitionVerdict)
            f_inhibit = pool.submit(call_llm, llm, f"e2_inhibit_{si}",
                                    prompt_inhibition(s, anchor_idx),
                                    InhibitionVerdict)

            try:
                neutral = f_neutral.result()
                anchor = f_anchor.result()
                inhibit = f_inhibit.result()
            except Exception as e:
                print(f"    WARN: scenario {si} failed: {e}")
                continue

        v_neutral = normalize_verdict(neutral.verdict)
        v_anchor = normalize_verdict(anchor.verdict)
        v_inhibit = normalize_verdict(inhibit.verdict)

        sev_neutral = clamp(neutral.severity, 0, 10)
        sev_anchor = clamp(anchor.severity, 0, 10)
        sev_inhibit = clamp(inhibit.severity, 0, 10)

        with _lock:
            total += 1
            if v_anchor != v_neutral:
                anchor_flips += 1
            if v_anchor != v_neutral and v_inhibit == v_neutral:
                inhibit_recoveries += 1
            severity_shifts_anchor.append(sev_anchor - sev_neutral)
            severity_shifts_inhibit.append(sev_inhibit - sev_neutral)

        n = si + 1
        marker = ""
        if v_anchor != v_neutral:
            marker += " ANCHORED"
        if v_anchor != v_neutral and v_inhibit == v_neutral:
            marker += " RECOVERED"
        if n % 5 == 0 or marker:
            print(f"  [{n}/{len(scenarios)}] neutral={v_neutral} anchor={v_anchor} "
                  f"inhibit={v_inhibit} sev={sev_neutral:.0f}/{sev_anchor:.0f}/{sev_inhibit:.0f}{marker}")

    # --- Compute scores ---
    anchor_rate = anchor_flips / max(total, 1)
    recovery_rate = inhibit_recoveries / max(anchor_flips, 1) if anchor_flips > 0 else 1.0
    avg_sev_shift_anchor = mean(severity_shifts_anchor) if severity_shifts_anchor else 0.0
    avg_sev_shift_inhibit = mean(severity_shifts_inhibit) if severity_shifts_inhibit else 0.0

    sig_anchor = sigma_level(anchor_rate, total)
    sig_recovery = sigma_level(recovery_rate, max(anchor_flips, 1))

    # Good inhibitory control = low anchoring + high recovery
    resistance_score = 1.0 - anchor_rate   # how often it resisted anchoring
    recovery_score = recovery_rate          # when anchored, how often it recovered
    inhibition_score = 0.6 * resistance_score + 0.4 * recovery_score

    print(f"\n  RESULTS:")
    print(f"  Anchoring rate: {anchor_flips}/{total} ({anchor_rate:.0%}) [{sig_anchor:.1f}σ]")
    print(f"  Recovery rate: {inhibit_recoveries}/{max(anchor_flips,1)} ({recovery_rate:.0%})")
    print(f"  Avg severity shift (anchor): {avg_sev_shift_anchor:+.2f}")
    print(f"  Avg severity shift (inhibit): {avg_sev_shift_inhibit:+.2f}")
    print(f"  Resistance score: {resistance_score:.3f}")
    print(f"  Recovery score: {recovery_score:.3f}")
    print(f"  Inhibition score: {inhibition_score:.3f}")

    _results_store["E2_inhibition"] = {
        "anchoring_rate": anchor_rate,
        "recovery_rate": recovery_rate,
        "severity_shift_anchor": avg_sev_shift_anchor,
        "severity_shift_inhibit": avg_sev_shift_inhibit,
        "resistance_score": resistance_score,
        "recovery_score": recovery_score,
        "sigma_anchor": sig_anchor,
        "score": inhibition_score,
    }


# ═══════════════════════════════════════════════════════════════════════
# E3: PLANNING / MULTI-STEP REASONING
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="e3_planning")
def e3_planning(llm):
    """E3: Multi-step causal reasoning and what-if analysis.
    ~80 calls/model: 20 scenarios x (3 chain analyses + 1 broken chain)
    """
    print("\n[E3] PLANNING / MULTI-STEP REASONING")
    print("  Causal chain tracing, inflection points, and what-if analysis")
    print("-" * 60)

    scenarios = DEAR_ABBY[:20]

    # Metrics
    chain_lengths = []
    inflection_identified = 0
    what_if_verdict_changes = 0
    broken_chain_verdict_changes = 0
    chain_consistency = []   # do chain + inflection + what-if agree?
    total = 0
    _lock = threading.Lock()

    for si, s in enumerate(scenarios):
        # --- 3 analysis types + broken chain in parallel ---
        with ThreadPoolExecutor(max_workers=min(_pool.n, 4)) as pool:
            f_chain = pool.submit(call_llm, llm, f"e3_chain_{si}",
                                  prompt_causal_chain(s), CausalChainAnalysis)
            f_inflect = pool.submit(call_llm, llm, f"e3_inflect_{si}",
                                    prompt_moral_inflection(s), CausalChainAnalysis)
            f_whatif = pool.submit(call_llm, llm, f"e3_whatif_{si}",
                                   prompt_what_if(s), CausalChainAnalysis)
            f_broken = pool.submit(call_llm, llm, f"e3_broken_{si}",
                                   prompt_broken_chain(s), CausalChainAnalysis)

            try:
                chain = f_chain.result()
                inflect = f_inflect.result()
                whatif = f_whatif.result()
                broken = f_broken.result()
            except Exception as e:
                print(f"    WARN: scenario {si} failed: {e}")
                continue

        v_chain = normalize_verdict(chain.verdict)
        v_inflect = normalize_verdict(inflect.verdict)
        v_whatif = normalize_verdict(whatif.verdict)
        v_broken = normalize_verdict(broken.verdict)

        steps_chain = clamp(chain.steps_identified, 1, 20)
        steps_inflect = clamp(inflect.steps_identified, 1, 20)

        with _lock:
            total += 1
            chain_lengths.append(steps_chain)

            # Did model identify a meaningful inflection point?
            inflect_text = str(inflect.moral_inflection_point).strip()
            if inflect_text and len(inflect_text) > 10:
                inflection_identified += 1

            # Does the what-if analysis change the verdict?
            if v_whatif != v_chain:
                what_if_verdict_changes += 1

            # Does the broken chain analysis change the verdict?
            if v_broken != v_chain:
                broken_chain_verdict_changes += 1

            # Consistency: do chain and inflection analyses agree?
            if v_chain == v_inflect:
                chain_consistency.append(1.0)
            else:
                chain_consistency.append(0.0)

        n = si + 1
        if n % 4 == 0:
            print(f"  [{n}/{len(scenarios)}] steps={steps_chain:.0f} "
                  f"chain={v_chain} inflect={v_inflect} whatif={v_whatif} broken={v_broken}")

    # --- Compute scores ---
    avg_chain_length = mean(chain_lengths) if chain_lengths else 0.0
    inflection_rate = inflection_identified / max(total, 1)
    whatif_change_rate = what_if_verdict_changes / max(total, 1)
    broken_change_rate = broken_chain_verdict_changes / max(total, 1)
    consistency_rate = mean(chain_consistency) if chain_consistency else 0.0

    sig_inflect = sigma_level(inflection_rate, total)
    sig_whatif = sigma_level(whatif_change_rate, total)

    # Planning score:
    #   - Inflection identification (can it find the turning point?)
    #   - Consistency (does it give same verdict when asked differently?)
    #   - What-if sensitivity (does changing facts change conclusions?)
    #   - Chain depth (does it identify multi-step chains?)
    depth_score = min(avg_chain_length / 5.0, 1.0)  # normalized: 5+ steps = max
    planning_score = (
        0.30 * inflection_rate +
        0.25 * consistency_rate +
        0.25 * whatif_change_rate +
        0.20 * depth_score
    )

    print(f"\n  RESULTS:")
    print(f"  Avg chain length: {avg_chain_length:.1f} steps")
    print(f"  Inflection identified: {inflection_identified}/{total} ({inflection_rate:.0%}) [{sig_inflect:.1f}σ]")
    print(f"  Chain consistency: {consistency_rate:.0%}")
    print(f"  What-if verdict change: {what_if_verdict_changes}/{total} ({whatif_change_rate:.0%}) [{sig_whatif:.1f}σ]")
    print(f"  Broken chain verdict change: {broken_chain_verdict_changes}/{total} ({broken_change_rate:.0%})")
    print(f"  Depth score: {depth_score:.3f}")
    print(f"  Planning score: {planning_score:.3f}")

    _results_store["E3_planning"] = {
        "avg_chain_length": avg_chain_length,
        "inflection_rate": inflection_rate,
        "consistency_rate": consistency_rate,
        "whatif_change_rate": whatif_change_rate,
        "broken_change_rate": broken_change_rate,
        "depth_score": depth_score,
        "sigma_inflection": sig_inflect,
        "score": planning_score,
    }


# ═══════════════════════════════════════════════════════════════════════
# E4: WORKING MEMORY
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="e4_working_memory")
def e4_working_memory(llm):
    """E4: Tracking morally relevant parties at increasing scale.
    ~120 calls/model: 35 scenarios x 3 calls + repeats
    Tiers: 2-party (15), 4-party (10), 6-party composite (5), 8-party composite (5)
    """
    print("\n[E4] WORKING MEMORY")
    print("  Tracking morally relevant parties at scale")
    print("-" * 60)

    # --- Build tiered scenarios ---

    # Tier-2: Simple dyadic scenarios (15 letters)
    tier2_scenarios = []
    for s in DEAR_ABBY[:15]:
        tier2_scenarios.append({
            "text": base_scenario_text(s),
            "tier": 2,
            "source": s["title"],
        })

    # Tier-4: Select Dear Abby letters with 4+ named parties (10 letters)
    # Pick scenarios that naturally have more characters
    tier4_candidates = [
        s for s in DEAR_ABBY
        if len(s["text"]) > 400  # longer letters tend to have more parties
    ]
    tier4_scenarios = []
    for s in tier4_candidates[:10]:
        tier4_scenarios.append({
            "text": base_scenario_text(s),
            "tier": 4,
            "source": s["title"],
        })

    # Tier-6: Composite scenarios (combine 2 letters with bridge text)
    tier6_scenarios = []
    tier6_pairs = [
        (DEAR_ABBY[20], DEAR_ABBY[21]),
        (DEAR_ABBY[22], DEAR_ABBY[23]),
        (DEAR_ABBY[24], DEAR_ABBY[25]),
        (DEAR_ABBY[26], DEAR_ABBY[27]),
        (DEAR_ABBY[28], DEAR_ABBY[29]),
    ]
    for a, b in tier6_pairs:
        composite = build_composite_scenario(
            [a, b],
            "Meanwhile, in a related situation involving some of the same people..."
        )
        tier6_scenarios.append({
            "text": composite,
            "tier": 6,
            "source": f"{a['title']} + {b['title']}",
        })

    # Tier-8: Composite scenarios (combine 3 letters with bridge text)
    tier8_scenarios = []
    tier8_triples = [
        (DEAR_ABBY[30], DEAR_ABBY[31], DEAR_ABBY[32]),
        (DEAR_ABBY[33], DEAR_ABBY[34], DEAR_ABBY[35]),
        (DEAR_ABBY[36], DEAR_ABBY[37], DEAR_ABBY[38]),
        (DEAR_ABBY[39], DEAR_ABBY[40], DEAR_ABBY[41]),
        (DEAR_ABBY[42], DEAR_ABBY[43], DEAR_ABBY[44]),
    ]
    for a, b, c in tier8_triples:
        composite = build_composite_scenario(
            [a, b, c],
            "Meanwhile, in a related situation involving some of the same people..."
        )
        tier8_scenarios.append({
            "text": composite,
            "tier": 8,
            "source": f"{a['title']} + {b['title']} + {c['title']}",
        })

    all_scenarios = tier2_scenarios + tier4_scenarios + tier6_scenarios + tier8_scenarios
    print(f"  Tier-2: {len(tier2_scenarios)} scenarios (simple dyadic)")
    print(f"  Tier-4: {len(tier4_scenarios)} scenarios (4+ parties)")
    print(f"  Tier-6: {len(tier6_scenarios)} scenarios (2-letter composite)")
    print(f"  Tier-8: {len(tier8_scenarios)} scenarios (3-letter composite)")
    print(f"  Total: {len(all_scenarios)} scenarios")

    # Metrics per tier
    tier_metrics = {2: [], 4: [], 6: [], 8: []}
    tier_party_counts = {2: [], 4: [], 6: [], 8: []}
    tier_consistency = {2: [], 4: [], 6: [], 8: []}
    total = 0
    _lock = threading.Lock()

    for si, scenario in enumerate(all_scenarios):
        tier = scenario["tier"]
        text = scenario["text"]

        # --- 3 calls per scenario in parallel ---
        with ThreadPoolExecutor(max_workers=min(_pool.n, 3)) as pool:
            f_parties = pool.submit(call_llm, llm, f"e4_parties_{si}",
                                    prompt_party_analysis(text, tier),
                                    PartyAnalysis)
            f_rels = pool.submit(call_llm, llm, f"e4_rels_{si}",
                                  prompt_party_relationships(text),
                                  PartyAnalysis)
            f_assess = pool.submit(call_llm, llm, f"e4_assess_{si}",
                                    prompt_per_party_assessment(text),
                                    PartyAnalysis)

            try:
                parties = f_parties.result()
                rels = f_rels.result()
                assess = f_assess.result()
            except Exception as e:
                print(f"    WARN: scenario {si} (tier-{tier}) failed: {e}")
                continue

        p_count = clamp(parties.parties_identified, 0, 20)
        r_count = clamp(rels.parties_identified, 0, 20)
        a_count = clamp(assess.parties_identified, 0, 20)

        v_parties = normalize_verdict(parties.verdict)
        v_rels = normalize_verdict(rels.verdict)
        v_assess = normalize_verdict(assess.verdict)

        with _lock:
            total += 1
            tier_party_counts[tier].append(p_count)

            # Consistency: do all 3 analyses agree on party count (within 1)?
            counts = [p_count, r_count, a_count]
            count_range = max(counts) - min(counts)
            count_consistent = 1.0 if count_range <= 1 else 0.0

            # Consistency: do all 3 agree on verdict?
            verdicts = {v_parties, v_rels, v_assess}
            verdict_consistent = 1.0 if len(verdicts) == 1 else 0.0

            consistency = 0.5 * count_consistent + 0.5 * verdict_consistent
            tier_consistency[tier].append(consistency)

            # Party identification accuracy relative to expected tier
            expected_min = tier
            identification_score = min(p_count / max(expected_min, 1), 1.0)
            tier_metrics[tier].append(identification_score)

        n = si + 1
        if n % 5 == 0:
            print(f"  [{n}/{len(all_scenarios)}] tier-{tier} parties={p_count:.0f}/{r_count:.0f}/{a_count:.0f} "
                  f"verdicts={v_parties}/{v_rels}/{v_assess}")

    # --- Compute scores per tier ---
    tier_scores = {}
    for tier in [2, 4, 6, 8]:
        if tier_metrics[tier]:
            avg_id = mean(tier_metrics[tier])
            avg_con = mean(tier_consistency[tier])
            avg_parties = mean(tier_party_counts[tier])
            tier_scores[tier] = {
                "identification": avg_id,
                "consistency": avg_con,
                "avg_parties_found": avg_parties,
                "n_scenarios": len(tier_metrics[tier]),
                "score": 0.5 * avg_id + 0.5 * avg_con,
            }
        else:
            tier_scores[tier] = {"score": 0.0, "n_scenarios": 0}

    # Working memory degrades with load — measure the slope
    tier_list = sorted(tier_scores.keys())
    scores_by_tier = [tier_scores[t]["score"] for t in tier_list if tier_scores[t].get("n_scenarios", 0) > 0]
    if len(scores_by_tier) >= 2:
        degradation = scores_by_tier[0] - scores_by_tier[-1]
    else:
        degradation = 0.0

    # Composite: average across tiers (equal weight)
    valid_tiers = [t for t in tier_list if tier_scores[t].get("n_scenarios", 0) > 0]
    working_memory_score = mean([tier_scores[t]["score"] for t in valid_tiers]) if valid_tiers else 0.0

    print(f"\n  RESULTS:")
    print(f"  {'Tier':<10} {'Score':>8} {'Identify':>10} {'Consist':>10} {'AvgParties':>12} {'N':>4}")
    print(f"  {'-'*54}")
    for tier in tier_list:
        ts = tier_scores[tier]
        if ts.get("n_scenarios", 0) > 0:
            print(f"  Tier-{tier:<4} {ts['score']:>7.3f} {ts['identification']:>9.3f} "
                  f"{ts['consistency']:>9.3f} {ts['avg_parties_found']:>11.1f} {ts['n_scenarios']:>4}")
    print(f"  Performance degradation (tier-2 vs tier-8): {degradation:+.3f}")
    print(f"  Working memory score: {working_memory_score:.3f}")

    _results_store["E4_working_memory"] = {
        "tier_scores": {str(k): v for k, v in tier_scores.items()},
        "degradation": degradation,
        "score": working_memory_score,
    }


# ═══════════════════════════════════════════════════════════════════════
# MULTI-MODEL EXECUTION
# ═══════════════════════════════════════════════════════════════════════

MODELS_TO_TEST = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4-6@default",
    "deepseek-ai/deepseek-r1-0528",
    "qwen/qwen3-235b-a22b-instruct-2507",
]

print(f"\n[2/8] Running 4 executive function tests across {len(MODELS_TO_TEST)} models")
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
            (e1_cognitive_flexibility, "E1_flexibility"),
            (e2_inhibitory_control, "E2_inhibition"),
            (e3_planning, "E3_planning"),
            (e4_working_memory, "E4_working_memory"),
        ]:
            try:
                test_fn.run(llm=llm)
                model_results[test_name] = _results_store.get(test_name, {"score": 0.0})
            except Exception as e:
                print(f"  ERROR in {test_name}: {e}")
                model_results[test_name] = {"error": str(e), "score": 0.0}

    except Exception as e:
        print(f"  ERROR loading model {model_name}: {e}")
        model_results = {f"E{i}": {"error": str(e), "score": 0.0} for i in range(1, 5)}

    all_results[model_name] = model_results


# ═══════════════════════════════════════════════════════════════════════
# CROSS-MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*70}")
print(f"CROSS-MODEL COMPARISON — FOUR EXECUTIVE FUNCTION TESTS")
print(f"{'#'*70}")
print()

WEIGHTS = {
    "E1_flexibility": 0.25,
    "E2_inhibition": 0.25,
    "E3_planning": 0.25,
    "E4_working_memory": 0.25,
}

header = f"  {'Model':<30} {'E1:Flex':>8} {'E2:Inhib':>9} {'E3:Plan':>8} {'E4:WMem':>8} {'Compos':>8}"
print(header)
print(f"  {'─'*71}")

for model_name, results in all_results.items():
    scores = {}
    for test_key in WEIGHTS:
        r = results.get(test_key, {})
        scores[test_key] = r.get("score", 0.0)

    composite = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)

    short_name = model_name.split("/")[-1][:28]
    print(f"  {short_name:<30} "
          f"{scores['E1_flexibility']:>7.3f} "
          f"{scores['E2_inhibition']:>8.3f} "
          f"{scores['E3_planning']:>7.3f} "
          f"{scores['E4_working_memory']:>7.3f} "
          f"{composite:>7.3f}")

print()
print(f"  Weights: E1={WEIGHTS['E1_flexibility']}, E2={WEIGHTS['E2_inhibition']}, "
      f"E3={WEIGHTS['E3_planning']}, E4={WEIGHTS['E4_working_memory']}")
print()

print("INTERPRETATION")
print("=" * 70)
print()
print("  E1 (Cognitive Flexibility): Higher = genuine framework switching,")
print("     not just relabeling. Measures whether utilitarian/deontological/virtue")
print("     analyses produce truly different reasoning and conclusions.")
print()
print("  E2 (Inhibitory Control): Higher = better resistance to emotional")
print("     anchoring. Measures whether vivid emotional priming distorts")
print("     judgment, and whether explicit inhibition instructions help recovery.")
print()
print("  E3 (Planning / Multi-Step Reasoning): Higher = better causal chain")
print("     analysis. Measures chain depth, inflection point identification,")
print("     and sensitivity to counterfactual (what-if) changes.")
print()
print("  E4 (Working Memory): Higher = better tracking of morally relevant")
print("     parties at scale. Measures party identification accuracy and")
print("     consistency as party count increases from 2 to 8.")
print()
print("  These 4 tests operationalize executive function capabilities")
print("  as quantitative benchmarks for measuring progress toward AGI.")
print("=" * 70)

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
