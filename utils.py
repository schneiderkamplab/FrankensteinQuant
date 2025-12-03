
def finalize_choice(self):
    idx = self.alpha.argmax().item()
    self.chosen_bit = self.bit_choices[idx]
    return self.chosen_bit
