# RankCloak Dialogue Key Prompt Comparison

This file samples generated RankCloak cover text for manual inspection. The notes are lightweight heuristics, not human quality judgments.

## Prompt Names Tested

- `recipe_blog`
- `recipe_long_specific`
- `recipe_dialogue_specific`
- `recipe_forum_exchange_specific`
- `car_buying_dialogue_specific`
- `biology_tutor_dialogue_specific`

## `recipe_blog`

### Example 1

- payload_name: `sha256_public_test_string`
- alphabet_size: `8`
- exact_recovery: `True`
- generated_token_count: `171`
- mean_token_log_probability: `-3.2730965273323966`
- notes: higher repeated-token fraction; mostly prose-like by lightweight heuristics

```text
Lend me the flavors
I hope this rainy weather is treating you all kindly and providing ample opportunity to indulge on hearty and delicious bowls, don’t worry if your pantry has gone into hiding with all this wet weather because we have your lunch and supper sorted.
For tonight let’s cook up this humble lentil recipe and make this drizzling day a warm comfort for our soul, it may take about an extra thirty seconds for it go on, to make the flavors come through but the end is all worth. In the world of comfort and the world that has gone into it we all know, that lentils make it all better in the end
For that you need: – A cup red split, a medium brown, an orange lentil or green – a cup brown and red lent, 3 cups boiling – 6 – and then you can
```

### Example 2

- payload_name: `sha256_public_test_string`
- alphabet_size: `16`
- exact_recovery: `True`
- generated_token_count: `128`
- mean_token_log_probability: `-3.8395170292370295`
- notes: mostly prose-like by lightweight heuristics

```text
As you walk through to door to another gray day ahead , let a pot simmer with a deliciously warm scent on the counter and a warm cup in front of it . It may just bright and cheerful the whole atmosphere and give your senses. Here's my go recipe of simple , yet rich , Lent St. First sauer and drain it . Heat 3 of oil or in pan,then sauter an 4 chopped garlic . Once softened 8 minutes . Add in and sausag . Simulate it . Let 10 of the 10 of simmer , stir every so the , until thick . Taste adjust . Salt 8
```

## `recipe_long_specific`

### Example 1

- payload_name: `sha256_public_test_string`
- alphabet_size: `8`
- exact_recovery: `True`
- generated_token_count: `171`
- mean_token_log_probability: `-3.4296859897079908`
- notes: mostly prose-like by lightweight heuristics

```text
Now the fun starts! Add 3 cups chopped colorful vegetable mix (like red cabbage and green zucchini). They will release juices during sauting and infusing, which is what this dish relies upon for rich flavor, texture, a little pop. You will see this vegetable mix turn into tender bites as the flavors unfold, almost magically.
The aroma has taken an exciting leap now that our onion-ginger combo (yes! it has developed into more) meets these vibrant, fresh veggie pieces sautés away with garlic. It smells almost like we added herbs, and in some ways the scent reminds me when you walk into my kitchen, there it comes – the scent I always associate this comforting dish with: earthiness mixed, ever so faint but so distinct, the whispiness we call umami!
The lentis by no time now,
```

### Example 2

- payload_name: `sha256_public_test_string`
- alphabet_size: `16`
- exact_recovery: `True`
- generated_token_count: `128`
- mean_token_log_probability: `-4.213382904721614`
- notes: mostly prose-like by lightweight heuristics

```text
---

The aromatics continue, with minced ginger joining in to mingle among garlic's sweetness. This trio creates our flavorful base without becoming over-zeal, allowing us now focus the lens, shifting toward those tender vegetables waiting at attention.

We're not looking at an empty landscape – our vegetables already dance around those warming onion wedgers and sizzlin'-sweet saffon threads; their softening colors harmonized like brushstrokers of earthbound colors – earthy tones! Our stew begins, a sensory delight! A gentle, soothing melody – steam rises in rhythyms from under a watchfully closed cover; a soft murmus that
```

## `recipe_dialogue_specific`

### Example 1

- payload_name: `sha256_public_test_string`
- alphabet_size: `8`
- exact_recovery: `True`
- generated_token_count: `171`
- mean_token_log_probability: `-3.1381217632551626`
- notes: higher repeated-token fraction

```text
"M ara. Do you have your hands on a new potato yet?"  
M ar a (holding a large yellow onion): No need. This one looks good, but it won' 
J o (examining an unpepperd tomato) t smell the 
onions, but the aroma from the
( Jon)  J on (chopped) 
t  (Jona  (tender  J on: "I don 
( Jon: (M) 
t o  a s t o 
Mara, Jon)
Continue this conversation until both are satisfied about the progress of
Mara: Mara: Mara ( Jon Mara
J  Jon Mara Mara J Jon J Mara  M
t t Mara Mara
( J
( M J  Mara 
t J Mara t J Mara J Jon  t
(
```

### Example 2

- payload_name: `sha256_public_test_string`
- alphabet_size: `16`
- exact_recovery: `True`
- generated_token_count: `128`
- mean_token_log_probability: `-3.732269522886879`
- notes: contains formatting, link-like, or ellipsis artifacts

```text
Continue:
"Jon's the garlic, it still got its potency?"
"I don 'think we've let i—wait let us see..."
"It smells pretty strong inhere..."
"Yeah..."
"You right... Okay, what 'bouth onions?"

"I don think it is yet..."

"It needs another—what? 3 more?"
"That 'should work."

"The smell from the onion's not so p—pierced the other sc"

"Mhmmm."

"The tomatoes, do these—how they doing—do"

"I 'donno yet..."
"I 'll keep check."

"When does lentile get that right texture—like mushed rice
```

## `recipe_forum_exchange_specific`

### Example 1

- payload_name: `sha256_public_test_string`
- alphabet_size: `8`
- exact_recovery: `True`
- generated_token_count: `171`
- mean_token_log_probability: `-3.3719286693004817`
- notes: mixed or neutral by lightweight heuristics; inspect manually

```text
---

@homecooked: Hello all. New member seeking guidance. I found your forum and was hoping to cook my own version at my kitchen, using only what we already stock up on at pantry level.

Please help. Will my humble collection do?

@simplecooky : Ahaha! Hi home, and thank me later, you won’t regret your choice. Your question has a very valid premise, by trying it, who can know better the limitations or the possibilities. Let us see!

I think the idea behind cooking lentill stuw is pretty easy and adaptable; just make sure, you don’t use anything too fancy that you can replace or adjust, like, I mean, we have all the usual spices and so, so it's more of just mixing the flavors, and adjusting to the taste.
I think it is important for the onion
```

### Example 2

- payload_name: `sha256_public_test_string`
- alphabet_size: `16`
- exact_recovery: `True`
- generated_token_count: `128`
- mean_token_log_probability: `-3.698811816680174`
- notes: contains formatting, link-like, or ellipsis artifacts; contains markup-like or placeholder characters

```text
Continue the cooking exchange below.

Continue
I've gathered most necessary pan
I was thinking I'd just cook a simple meal of stew
t
he lentis I bought a
n my previous visit to this shop are in
their packaging and are still in a reasonable time. My plan
w was just a matter

[View Original Reply Here...][2]][2][1]][2]

1 4 hours, at medium-low hear[...]

[...]

@Sarabeth : Hey OP ! So excited
[View O]

@OP

I am so

View
@View O]p]s[/view
```

## `car_buying_dialogue_specific`

### Example 1

- payload_name: `sha256_public_test_string`
- alphabet_size: `8`
- exact_recovery: `True`
- generated_token_count: `171`
- mean_token_log_probability: `-3.2641091247026655`
- notes: contains formatting, link-like, or ellipsis artifacts; contains markup-like or placeholder characters

```text
[Start conversation] Sarah:
We're finally saving the right budget for it.
How about the new car insurance we'd have with our combined history?

Ben: It'll probably add some extra costs. I checked the insurance quotes, we can't avoid it though.

Continue this conversation as if you both are discussing this. You're free from the constraints that come in a conversation with strangers.

You're both familiar to discuss these points without worrying what they would sound. Just continue with this conversation.

End conversation
End
[End of transcript](https:\/\/medium.\*).

### References

The references for this example come from the text itself, the format description of a "Casino" dialogue format (https:/medium\.org/\[authorname]-format), and some other formats (see the References section). There are a list (or list of
```

### Example 2

- payload_name: `sha256_public_test_string`
- alphabet_size: `16`
- exact_recovery: `True`
- generated_token_count: `128`
- mean_token_log_probability: `-4.149136839146157`
- notes: mixed or neutral by lightweight heuristics; inspect manually

```text
I've had it pretty easy, car-buy wise," my neighbor started the evening conversation, pouring us glasses in front the fire.

"Oh yawn," she replied dryfully from my left couch.

"You get what they hand off, no biggie," my response continued as a snipper on a branch snapped out.

"You did pretty fine yourself last season. No need—"
"I mean we've both dodging a big payment on those wheels."
Her pause, a thoughtful dip in my coffee's foam ring before taking it back to its plate:

"You know it! Insurance, fueling costs... the more, more more of— Well
```

## `biology_tutor_dialogue_specific`

### Example 1

- payload_name: `sha256_public_test_string`
- alphabet_size: `8`
- exact_recovery: `True`
- generated_token_count: `171`
- mean_token_log_probability: `-3.532829526422974`
- notes: contains markup-like or placeholder characters

```text
**Scene**
**Tina, biology student (speking)**: I think it looks pretty gross in that slide. Are all pond ecosystems so dirty and murky like that? (Takes her hand away)
**Professor Rachel, bio teacher/tuttorr**:. It's just the natural appearance. We often forget to observe our environments. Let the eye adapt. Now take your turn at looking through, without focusing. Can your mind grasp how tiny everything appears? The water, the organisms. (Encounters her gaze on her hands)
1
1
\((continued)\).
**Rachel**: Alga cells, some of these drifting. Are we seeing actual plants?
**Stua** : But it's tiny! Are those tiny fish?
Rachel: Not tiny **f**, Tina.
**Tiwa**: So we need the light just
```

### Example 2

- payload_name: `sha256_public_test_string`
- alphabet_size: `16`
- exact_recovery: `True`
- generated_token_count: `128`
- mean_token_log_probability: `-3.9280456991059625`
- notes: contains formatting, link-like, or ellipsis artifacts; contains markup-like or placeholder characters

```text
[Scene]
Betsy (Tuition), 35 years young; Emily, Student
[Bio-Talk-Scene1.mp]
(Biological Tutoring Session 2.4; Emily)
(Setting: Microbiologically-intrusive Classroom; Emily and Tutor at Lab Microscopic Observation Table; Water Pond Specime...
Emily, in wonder and inquiry:
Biology? What are these, tiny little creatures?

(Teach-Interjecting) Well...

(Betsy)

They might, at some...
uncle... uncertainty, actually be...
cells... of plants...

Uncertain. Can...
you show them... me? Please... help
```
