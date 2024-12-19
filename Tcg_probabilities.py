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
def make_deck(evs, sp):
    deck = []
    for line in evs:
        for card, count in line.items():
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
        while "pball" in hand:
            s = sample_from_basics(deck, basics, 1)
            bas, deck = s[0], s[1]
            hand.remove("pball")
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
    # Generate combinations dynamically based on the input evs list
    from itertools import chain, combinations

    def all_combinations(evs):
        evs_sets = [set(group) for group in evs]
        all_combos = chain.from_iterable(combinations(evs_sets, r) for r in range(1, len(evs_sets) + 1))
        return {
            ' + '.join('[' + ', '.join(group) + ']' for group in combo): set.union(*combo)
            for combo in all_combos
        }

    combinations_dict = all_combinations(evs)

    # Initialize a dictionary to store counts for each combination by turn
    counts = {key: [0] * max_turn for key in combinations_dict.keys()}

    # Additional categories for EVS of specific lengths
    at_least_one_len_2 = [0] * max_turn
    at_least_one_len_3 = [0] * max_turn

    # Iterate through all simulations
    for sim_id, hands in Hands.items():
        # Iterate through each turn
        for turn in range(max_turn):
            if turn in hands:
                hand_set = set(hands[turn])
                # Check for each combination
                for key, combo in combinations_dict.items():
                    if combo.issubset(hand_set):
                        counts[key][turn] += 1
                # Check for at least one EVS of specific lengths
                for evs_group in evs:
                    evs_set = set(evs_group.keys())
                    if evs_set.issubset(hand_set):
                        if len(evs_group) == 2:
                            at_least_one_len_2[turn] += 1
                        elif len(evs_group) == 3:
                            at_least_one_len_3[turn] += 1

    # Convert counts to probabilities
    tot_sims = len(Hands)
    probabilities = {key: [count / tot_sims for count in counts[key]] for key in counts.keys()}
    probabilities["At least one 2-pokemon evolution line"] = [count / tot_sims for count in at_least_one_len_2]
    probabilities["At least one 3-pokemon evolution line"] = [count / tot_sims for count in at_least_one_len_3]

    return probabilities


def plot_all_probabilities(probabilities, max_turn):
    plt.figure(figsize=(15, 10))
    for key, probs in probabilities.items():
        x = range(max_turn)
        y = [p * 100 for p in probs]  # Convert probabilities to percentages
        plt.plot(x, y, label=key, linestyle='-', marker='o')
        
        # Add labels for each point
        for xi, yi in zip(x, y):
            plt.text(xi, yi + 1, f"{yi:.1f}%", ha="center", fontsize=8)  # Offset for clarity

    plt.xlabel('Turn')
    plt.ylabel('Probability (%)')
    plt.title('Probabilities of EVS Combinations Over Turns')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt
# Streamlit Interface
import streamlit as st
st.title("EVS Probability Simulation")

# Input: Number of Evolution Lines
st.header("Number of Evolution Lines")
num_evs = st.number_input("Enter number of evolution lines", min_value=1, value=3, step=1)

# Input: Evolution Lines
st.header("Enter Evolution Lines and Quantities")
evs = []
for i in range(1, num_evs + 1):
    st.subheader(f"Evolution Line {i}")
    cards = st.text_input(f"Enter cards for line {i} (comma-separated):", key=f"line_{i}")
    quantities = st.text_input(f"Enter quantities for line {i} (comma-separated, same order):", key=f"quantities_{i}")
    if cards and quantities:
        card_list = cards.split(",")
        quantity_list = [int(x) for x in quantities.split(",")]
        evs.append(dict(zip(card_list, quantity_list)))

# Input: Spells
st.header("Enter Spell Cards and Quantities")
sp_cards = st.text_input("Enter spell cards (comma-separated) (IMPORTANT: use 'pball' and 'oak'!!):")
sp_quantities = st.text_input("Enter spell quantities (comma-separated, same order):")
sp = {}
if sp_cards and sp_quantities:
    sp_card_list = sp_cards.split(",")
    sp_quantity_list = [int(x) for x in sp_quantities.split(",")]
    sp = dict(zip(sp_card_list, sp_quantity_list))

# Determine Basics
basics = [list(line.keys())[0] for line in evs]

# Input: Number of Simulations
num_simulations = st.number_input("Enter Number of Simulations", min_value=1, value=10000, step=1)

# Run Simulation Button
if st.button("Run Simulation"):
    deck = make_deck(evs, sp)
    total_cards = len(deck)
    if total_cards != 20:
        st.error(f"Invalid deck size: {total_cards}. Deck must contain exactly 20 cards.")
    else:
        Hands = run_N_hands(deck, basics, num_simulations)
        max_turn = 9
        probabilities = compute_probabilities(Hands, evs, max_turn)

        # Select which curves to include before generating the graph
        st.header("Select Curves to Include")
        if "curves_to_plot" not in st.session_state:
            st.session_state["curves_to_plot"] = {key: True for key in probabilities.keys()}

        for key in probabilities.keys():
            st.session_state["curves_to_plot"][key] = st.checkbox(f"Include {key}", value=st.session_state["curves_to_plot"][key])

        # Generate Graph Button
        if st.button("Generate Graph"):
            plt.figure(figsize=(15, 10))

            # Check if any curve is selected
            selected_curves = [key for key, show in st.session_state["curves_to_plot"].items() if show]

            if selected_curves:
                for key in selected_curves:
                    x = range(max_turn)
                    y = [p * 100 for p in probabilities[key]]  # Convert probabilities to percentages
                    plt.plot(x, y, label=key, linestyle='-', marker='o')

                    # Add labels for each point
                    for xi, yi in zip(x, y):
                        plt.text(xi, yi + 1, f"{yi:.1f}%", ha="center", fontsize=8)  # Offset for clarity

                plt.xlabel('Turn')
                plt.ylabel('Probability (%)')
                plt.title('Probabilities of EVS Combinations Over Turns')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(plt.gcf())
            else:
                # Show a placeholder graph when no curves are selected
                st.warning("No curves selected. Displaying an empty graph.")
                plt.figure(figsize=(15, 10))
                plt.xlabel('Turn')
                plt.ylabel('Probability (%)')
                plt.title('No Curves Selected')
                plt.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(plt.gcf())

        # Save Option
        save_plot = st.checkbox("Save Plot as PNG")
        if save_plot:
            plt.savefig("evs_probabilities.png")
            st.success("Plot saved as 'evs_probabilities.png'.")
