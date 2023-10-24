
import isr_spec

# create 2-ion spectra for O_2+/N_2+ and O+
isr_spec.il_table(mass0=31,mass1=16.0)
# create 2-ion spectra for O+ and H+
isr_spec.il_table(mass0=16.0,mass1=1.0)
