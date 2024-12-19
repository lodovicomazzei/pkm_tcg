#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:52:52 2024

@author: mazzei
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain, combinations
import random

# Function Definitions
def make_deck(pk, sp):
    deck = []
    for card, count in pk.items():
        deck.extend([card] * count)
    for card, count in sp.items():
        deck.extend([card] * count)
    return deck

def sample_from_basics(input_list, basics, n):
    valid_indices = [i for i, val in enumerate(input_list) if val in basics]
    if not valid_indices:
        return [], input_list
    if n > len(valid_indices):
        raise ValueError("n cannot be greater than the number of valid elements in input_list")
    random.shuffle(valid_indices)
    sampled_indices = valid_indices[:n]
    sampled_values = [input_list[i] for i in sampled_indices]
    remaining_list = [input_list[i] for i in range(len(input_list)) if i not in sampled_indices]
    return sampled_values, remaining_list

def sample_without_replacement(input_list, n):
    if n >= len(input_list):
        return input_list, []
    indices = list(range(len(input_list)))
    random.shuffle(indices)
    sampled_indices = indices[:n]
    sampled_values = [input_list[i] for i in sampled_indices]
    remaining_list = [input_list[i] for i in range(len(input_list)) if i not in sampled_indices]
    return sampled_values, remaining_list

def simulate_draw(deck, basics):
    hands = {}
    s = sample_from_basics(deck, basics, 1)
    first_basic, deck = s[0], s[1]
    s = sample_without_replacement(deck, 4)
    other_4, deck = s[0], s[1]
    hands[0] = first_basic + other_4
    t = 0
    while deck:
        s = sample_without_replacement(deck, 1)
        card, deck = s[0], s[1]
        hand = hands[t] + card
        while "pob" in hand:
            s = sample_from_basics(deck, basics, 1)
            bas, deck = s[0], s[1]
            hand.remove("pob")
            hand += bas
        if "oak" in hand:
            s = sample_without_replacement(deck, 2)
            pair, deck = s[0], s[1]
            hand.remove("oak")
            hand += pair
        t += 1
        hands[t] = hand
    return hands

def run_N_hands(deck, basics, N):
    Hands = {}
    for i in range(N):
        hands = simulate_draw(deck, basics)
        Hands[i] = hands
    return Hands

def compute_probabilities(Hands, evs, max_turn):
    evs_sets = [set(group) for group in evs]
    all_combos = chain.from_iterable(combinations(evs_sets, r) for r in range(1, len(evs_sets) + 1))
    combinations = {
        ' + '.join('[' + ', '.join(group) + ']' for group in combo): set.union(*combo)
        for combo in all_combos
    }
    counts = {key: [0] * max_turn for key in combinations.keys()}
    for hands in Hands.values():
        for turn in range(max_turn):
            if turn in hands:
                hand_set = set(hands[turn])
                for key, combo in combinations.items():
                    if combo.issubset(hand_set):
                        counts[key][turn] += 1
    tot_sims = len(Hands)
    probabilities = {key: [count / tot_sims for count in counts[key]] for key in counts.keys()}
    return probabilities

def plot_all_probabilities(probabilities, max_turn):
    plt.figure(figsize=(15, 10))
    for key, probs in probabilities.items():
        x = range(max_turn)
        y = [p * 100 for p in probs]
        plt.plot(x, y, label=key)
        for xi, yi in zip(x, y):
            plt.scatter(xi, yi, color='black')
            plt.text(xi, yi + 1, f'{yi:.1f}%', ha='center', fontsize=8)
    plt.xlabel('Turn')
    plt.ylabel('Probability (%)')
    plt.title('Probabilities of EVS Combinations Over Turns')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt

# Streamlit Interface
st.title("EVS Probability Simulation")

# Input: EVS Lines
evs_input = st.text_area("Enter EVS Lines (e.g., [['kof', 'wez'], ['ven', 'whi', 'sco'], ['mew']])", "[['kof', 'wez'], ['ven', 'whi', 'sco'], ['mew']]")
evs = eval(evs_input)

# Input: SP Cards
sp_input = st.text_area("Enter SP Cards with counts (e.g., {'pob': 2, 'oak': 2})", "{'pob': 2, 'oak': 2}")
sp = eval(sp_input)

# Input: Number of Simulations
num_simulations = st.number_input("Enter Number of Simulations", min_value=1, value=10000, step=1)

# Run Simulation Button
if st.button("Run Simulation"):
    pk = {card: 2 for card in [group[0] for group in evs]}
    deck = make_deck(pk, sp)
    basics = [group[0] for group in evs]
    Hands = run_N_hands(deck, basics, num_simulations)
    max_turn = 9
    probabilities = compute_probabilities(Hands, evs, max_turn)
    
    # Plot and Show Results
    plt = plot_all_probabilities(probabilities, max_turn)
    st.pyplot(plt.gcf())
    
    # Save Option
    save_plot = st.checkbox("Save Plot as PNG")
    if save_plot:
        plt.savefig("evs_probabilities.png")
        st.success("Plot saved as 'evs_probabilities.png'.")
